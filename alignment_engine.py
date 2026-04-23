import os
import torch
import soundfile as sf
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from typing import List, Dict

class AlignmentEngine:
    def __init__(self, whisper_path: str, wav2vec2_path: str):
        self.whisper_path = whisper_path
        self.wav2vec2_path = wav2vec2_path
        self.whisper_model = None
        self.whisper_processor = None
        self.wav2vec2_model = None
        self.wav2vec2_processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_models(self):
        # Verification before loading
        whisper_bin = os.path.join(self.whisper_path, "pytorch_model.bin")
        wav2vec2_bin = os.path.join(self.wav2vec2_path, "pytorch_model.bin")
        
        if not os.path.exists(whisper_bin):
            raise FileNotFoundError(f"Whisper model weights not found at {whisper_bin}. Please run model_downloader.py.")
        
        if not os.path.exists(wav2vec2_bin):
            raise FileNotFoundError(f"Wav2Vec2 model weights not found at {wav2vec2_bin}. Please run model_downloader.py.")

        print(f"Loading Whisper model from {self.whisper_path}...")
        try:
            self.whisper_processor = WhisperProcessor.from_pretrained(self.whisper_path)
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(self.whisper_path).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}")
        
        print(f"Loading Wav2Vec2 model from {self.wav2vec2_path}...")
        try:
            self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained(self.wav2vec2_path)
            self.wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(self.wav2vec2_path).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load Wav2Vec2 model: {e}")

    def transcribe(self, audio_path: str) -> str:
        if not self.whisper_model:
            self.load_models()
            
        speech, sr = sf.read(audio_path, dtype='float32')
        # Convert to mono if stereo
        if len(speech.shape) > 1:
            speech = speech.mean(axis=1)
            
        if sr != 16000:
            import librosa
            speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
            sr = 16000
            
        input_features = self.whisper_processor(speech, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
        
        predicted_ids = self.whisper_model.generate(input_features)
        transcription = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription

    def align(self, audio_path: str, reference_text: str) -> List[Dict]:
        """Performs forced alignment to get word-level timestamps."""
        if not self.wav2vec2_model:
            self.load_models()
            
        speech, sr = sf.read(audio_path, dtype='float32')
        # Convert to mono if stereo
        if len(speech.shape) > 1:
            speech = speech.mean(axis=1)
            
        if sr != 16000:
            import librosa
            speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        audio_tensor = torch.from_numpy(speech).to(self.device)
        
        # Get emissions
        with torch.inference_mode():
            logits = self.wav2vec2_model(audio_tensor.unsqueeze(0)).logits
            emissions = torch.log_softmax(logits, dim=-1)
        
        emission = emissions[0].cpu().detach()
        
        # Build tokens array from reference text
        words = reference_text.split()
        vocab = self.wav2vec2_processor.tokenizer.get_vocab()
        special_ids = [vocab.get('[PAD]', -1), vocab.get('[UNK]', -1), vocab.get('<s>', -1), vocab.get('</s>', -1)]
        word_delimiter = getattr(self.wav2vec2_processor.tokenizer, 'word_delimiter_token', None)
        word_delimiter_id = vocab.get(word_delimiter, -1) if word_delimiter else -1
        
        # Strategy 1: Encode the full text to preserve context
        full_tokens = self.wav2vec2_processor.tokenizer.encode(reference_text)
        full_tokens = [t for t in full_tokens if t not in special_ids]
        
        word_token_spans = []
        if word_delimiter_id != -1:
            start_idx = 0
            for idx, t in enumerate(full_tokens):
                if t == word_delimiter_id:
                    if start_idx < idx:
                        word_token_spans.append((start_idx, idx - 1))
                    start_idx = idx + 1
            if start_idx < len(full_tokens):
                word_token_spans.append((start_idx, len(full_tokens) - 1))
        
        # Strategy 2: Fallback to word-by-word if spans don't match word count
        if len(word_token_spans) != len(words):
            full_tokens = []
            word_token_spans = []
            for i, word in enumerate(words):
                word_tokens = self.wav2vec2_processor.tokenizer.encode(word)
                word_tokens_filtered = [t for t in word_tokens if t not in special_ids and t != word_delimiter_id]
                
                if not word_tokens_filtered:
                    word_tokens_filtered = [vocab.get('[UNK]', 3)]
                
                start_idx = len(full_tokens)
                full_tokens.extend(word_tokens_filtered)
                end_idx = len(full_tokens) - 1
                word_token_spans.append((start_idx, end_idx))
                
                if i < len(words) - 1 and word_delimiter_id != -1:
                    full_tokens.append(word_delimiter_id)
        
        tokens = full_tokens
        if not tokens:
            return []
            
        # CTC Alignment logic
        trellis = self._get_trellis(emission, tokens)
        path = self._backtrack(trellis, emission, tokens)
        
        # Filter path to get only the transition frames (start of each token)
        transition_frames = [p for p in path if p.get('is_changed')]
        
        # If backtracking failed or returned too few transitions
        if len(transition_frames) < len(tokens):
            # Fallback: Just spread tokens evenly across the audio
            step = emission.shape[0] / max(len(tokens), 1)
            transition_frames = [{'time_index': int(i * step), 'token_index': i} for i in range(len(tokens))]

        word_alignments = []
        ratio = len(audio_tensor) / 16000 / emission.shape[0]
        total_duration = len(audio_tensor) / 16000
        
        for i, word in enumerate(words):
            start_idx, end_idx = word_token_spans[i]
            
            # Start time is the start of the first token of the word
            start_frame = transition_frames[start_idx]['time_index']
            start_time = round(start_frame * ratio, 3)
            
            # The token that immediately follows the word's last token
            # determines when the word's last token ends.
            if end_idx + 1 < len(transition_frames):
                end_frame = transition_frames[end_idx + 1]['time_index']
                end_time = round(end_frame * ratio, 3)
                
                if i < len(words) - 1:
                    next_start_idx = word_token_spans[i + 1][0]
                    next_start_time = round(transition_frames[next_start_idx]['time_index'] * ratio, 3)
                    
                    gap = next_start_time - end_time
                    if gap > 0:
                        # There is a delimiter gap. Add a tiny buffer to capture trailing breath/sound
                        end_time += min(gap, 0.05)
                    else:
                        # Contiguous (no delimiter). Pull back slightly to prevent bleeding!
                        end_time -= 0.03
                else:
                    end_time += 0.05
            else:
                # Last word and no tokens follow it
                end_frame = transition_frames[end_idx]['time_index']
                end_time = round((end_frame + 10) * ratio, 3)  # approx 200ms
            
            # Ensure boundaries are logical
            if end_time > total_duration:
                end_time = total_duration
                
            if end_time <= start_time:
                end_time = start_time + 0.1
                
            word_alignments.append({
                "word": word,
                "start": round(start_time, 3),
                "end": round(end_time, 3),
                "confidence": 0.9  # Fixed confidence for simplicity as it's not strictly needed
            })
        
        # Log alignments for debugging
        print(f"[Alignment] Generated {len(word_alignments)} word timestamps:")
        for wa in word_alignments:
            print(f"  {wa['word']}: {wa['start']}s - {wa['end']}s (conf: {wa['confidence']})")
        
        return word_alignments

    def align_whisper(self, audio_path: str, reference_text: str) -> List[Dict]:
        """Uses Whisper's built-in word-level timestamps via cross-attention weights.
        
        Uses model.generate() with return_token_timestamps=True directly,
        avoiding the pipeline (which requires torchaudio/ffmpeg).
        """
        if not self.whisper_model:
            self.load_models()
        
        # Load audio with soundfile (no ffmpeg needed)
        print(f"[Whisper] Processing {audio_path} for word timestamps...")
        speech, sr = sf.read(audio_path, dtype='float32')
        if len(speech.shape) > 1:
            speech = speech.mean(axis=1)
        if sr != 16000:
            import librosa
            speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Process audio with attention_mask for precise timestamps
        inputs = self.whisper_processor(
            speech, sampling_rate=16000, return_tensors="pt",
            return_attention_mask=True
        )
        input_features = inputs.input_features.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device) if hasattr(inputs, 'attention_mask') and inputs.attention_mask is not None else None
        
        # Generate with token-level timestamps from cross-attention weights
        # Note: language/task args removed - this model is already Arabic Quran-specific
        # Set alignment_heads from openai/whisper-base (required for token timestamps)
        if not hasattr(self.whisper_model.generation_config, 'alignment_heads') or \
           self.whisper_model.generation_config.alignment_heads is None:
            # Alignment heads for whisper-base architecture
            self.whisper_model.generation_config.alignment_heads = [
                [3, 1], [4, 2], [4, 7], [5, 1], [5, 2], [5, 4]
            ]
            print("[Whisper] Set alignment_heads from whisper-base config")
        
        try:
            generate_kwargs = {
                "return_token_timestamps": True,
                "return_dict_in_generate": True,
            }
            if attention_mask is not None:
                generate_kwargs["attention_mask"] = attention_mask
            
            result = self.whisper_model.generate(
                input_features,
                **generate_kwargs,
            )
        except Exception as e:
            print(f"[Whisper] Generation failed: {e}, falling back to CTC")
            return self.align(audio_path, reference_text)
        
        # Debug: print what we got back
        print(f"[Whisper] Generate result type: {type(result).__name__}")
        if hasattr(result, 'keys'):
            print(f"[Whisper] Result keys: {list(result.keys())}")
        
        # Extract token IDs - handle multiple return formats
        if hasattr(result, 'sequences'):
            token_ids = result.sequences[0]
        elif isinstance(result, torch.Tensor):
            token_ids = result[0]
        else:
            # Dict-like object, find sequences
            for key in ['sequences', 'output_token_ids']:
                if key in result:
                    token_ids = result[key][0]
                    break
            else:
                print(f"[Whisper] ERROR: Cannot find token IDs in result")
                return self.align(audio_path, reference_text)
        
        # Extract timestamps
        token_timestamps = None
        if hasattr(result, 'token_timestamps') and result.token_timestamps is not None:
            token_timestamps = result.token_timestamps[0]
        elif isinstance(result, dict) and 'token_timestamps' in result:
            token_timestamps = result['token_timestamps'][0]
        
        if token_timestamps is None:
            print("[Whisper] WARNING: token_timestamps not available, falling back to CTC")
            return self.align(audio_path, reference_text)
        
        # Decode full transcription
        transcription = self.whisper_processor.decode(token_ids, skip_special_tokens=True)
        print(f"[Whisper] Transcribed: {transcription}")
        
        # Group tokens into words with timestamps
        tokenizer = self.whisper_processor.tokenizer
        whisper_words = []
        current_word_tokens = []
        current_word_start = None
        current_word_end = None
        
        for i, (tid, ts) in enumerate(zip(token_ids, token_timestamps)):
            # Skip special tokens
            if tid.item() in tokenizer.all_special_ids:
                continue
            
            token_text = tokenizer.decode([tid.item()])
            ts_val = ts.item()
            
            # Skip timestamp tokens (Whisper uses special <|0.00|> tokens)
            if token_text.startswith("<|") and token_text.endswith("|>"):
                continue
            
            # Check if this token starts a new word (has leading space)
            if token_text.startswith(" ") or token_text.startswith("Ġ"):
                # Save previous word if exists
                if current_word_tokens:
                    word_text = tokenizer.decode(current_word_tokens).strip()
                    if word_text:
                        whisper_words.append({
                            "text": word_text,
                            "start": current_word_start,
                            "end": current_word_end
                        })
                current_word_tokens = [tid.item()]
                current_word_start = ts_val
                current_word_end = ts_val
            elif not current_word_tokens:
                # First token of first word (no leading space)
                current_word_tokens = [tid.item()]
                current_word_start = ts_val
                current_word_end = ts_val
            else:
                # Continuation of current word
                current_word_tokens.append(tid.item())
                current_word_end = ts_val
        
        # Don't forget the last word
        if current_word_tokens:
            word_text = tokenizer.decode(current_word_tokens).strip()
            if word_text:
                whisper_words.append({
                    "text": word_text,
                    "start": current_word_start,
                    "end": current_word_end
                })
        
        print(f"[Whisper] Got {len(whisper_words)} words from Whisper")
        for w in whisper_words:
            print(f"  '{w['text']}': {w['start']}s - {w['end']}s")
        
        if not whisper_words:
            return []
        
        # Map Whisper words to reference text words
        ref_words = reference_text.split()
        word_alignments = []
        
        if len(whisper_words) == len(ref_words):
            # Perfect match: use Whisper timestamps with reference text
            print("[Whisper] Word count matches reference - direct mapping")
            for i, ww in enumerate(whisper_words):
                word_alignments.append({
                    "word": ref_words[i],
                    "start": round(ww["start"], 3),
                    "end": round(ww["end"], 3),
                    "confidence": 0.95
                })
        else:
            # Counts differ: use smart matching
            print(f"[Whisper] Word count mismatch: Whisper={len(whisper_words)}, Reference={len(ref_words)}")
            chunks = [{"timestamp": (w["start"], w["end"]), "text": w["text"]} for w in whisper_words]
            word_alignments = self._match_whisper_to_reference(chunks, ref_words)
        
        # Adjust timestamps to prevent bleeding and preserve natural gaps
        for i in range(len(word_alignments) - 1):
            gap = word_alignments[i + 1]["start"] - word_alignments[i]["end"]
            if gap > 0:
                # Add a small buffer to the end of the current word to catch trailing sounds
                word_alignments[i]["end"] += min(gap, 0.05)
            elif gap < 0:
                # Overlap! Fix it by setting end to next start minus a tiny margin
                word_alignments[i]["end"] = word_alignments[i + 1]["start"] - 0.02
            else:
                # Contiguous. Pull back slightly to prevent bleeding into next word
                word_alignments[i]["end"] -= 0.03
                
            # Ensure start is strictly before end
            if word_alignments[i]["end"] <= word_alignments[i]["start"]:
                word_alignments[i]["end"] = word_alignments[i]["start"] + 0.05
            
            # Round values
            word_alignments[i]["end"] = round(word_alignments[i]["end"], 3)
        
        # Log final alignments
        print(f"[Whisper] Generated {len(word_alignments)} word timestamps:")
        for wa in word_alignments:
            print(f"  {wa['word']}: {wa['start']}s - {wa['end']}s (conf: {wa['confidence']})")
        
        return word_alignments

    def _match_whisper_to_reference(self, whisper_chunks: List[Dict], ref_words: List[str]) -> List[Dict]:
        """Match Whisper word chunks to reference text words when counts differ.
        
        Uses the Whisper timeline but distributes it across reference words
        proportionally based on character length.
        """
        # Get the full time range from Whisper
        total_start = whisper_chunks[0]["timestamp"][0]
        total_end = whisper_chunks[-1]["timestamp"][1]
        
        if len(whisper_chunks) > len(ref_words):
            # Whisper has more words - merge some chunks to match reference count
            # Distribute chunks across reference words proportionally
            chunks_per_word = len(whisper_chunks) / len(ref_words)
            word_alignments = []
            
            for i in range(len(ref_words)):
                start_chunk = int(round(i * chunks_per_word))
                end_chunk = int(round((i + 1) * chunks_per_word)) - 1
                end_chunk = min(end_chunk, len(whisper_chunks) - 1)
                start_chunk = min(start_chunk, len(whisper_chunks) - 1)
                
                start_time = whisper_chunks[start_chunk]["timestamp"][0]
                end_time = whisper_chunks[end_chunk]["timestamp"][1]
                
                word_alignments.append({
                    "word": ref_words[i],
                    "start": round(start_time, 3),
                    "end": round(end_time, 3),
                    "confidence": 0.8
                })
            
            return word_alignments
        else:
            # Whisper has fewer words - split time proportionally by char length
            char_lengths = [max(len(w), 1) for w in ref_words]
            total_chars = sum(char_lengths)
            total_duration = total_end - total_start
            
            word_alignments = []
            current_time = total_start
            
            for i, word in enumerate(ref_words):
                proportion = char_lengths[i] / total_chars
                word_duration = total_duration * proportion
                word_start = current_time
                word_end = current_time + word_duration
                
                word_alignments.append({
                    "word": word,
                    "start": round(word_start, 3),
                    "end": round(word_end, 3),
                    "confidence": 0.7
                })
                current_time = word_end
            
            return word_alignments

    def _get_trellis(self, emission, tokens, blank_id=0):
        num_frame = emission.size(0)
        num_tokens = len(tokens)
        trellis = torch.zeros((num_frame, num_tokens + 1))
        trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
        trellis[0, 1:] = -float('inf')
        trellis[-num_tokens:, 0] = float('inf')
        
        for t in range(num_frame - 1):
            trellis[t + 1, 1:] = torch.maximum(
                trellis[t, 1:] + emission[t, blank_id],
                trellis[t, :-1] + emission[t, tokens],
            )
        return trellis

    def _backtrack(self, trellis, emission, tokens, blank_id=0):
        t, j = trellis.size(0) - 1, trellis.size(1) - 1
        path = []
        while j > 0 and t > 0:
            stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
            changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]
            
            path.append({
                'time_index': t - 1,
                'token_index': j - 1,
                'is_changed': changed > stayed
            })
            
            t -= 1
            if changed > stayed:
                j -= 1
                
        return path[::-1]




if __name__ == "__main__":
    pass
