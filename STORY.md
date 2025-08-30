# Sentinel: Multimodal AI Privacy Guardian for Social Media

## 1. Background – Why Privacy-by-Default Now

Data protection has shifted from a compliance checkbox to a frontline user expectation. Modern AI systems ingest massive volumes of unstructured, user-generated content: photos, screenshots, chat logs, captions, comments. Even *well‑intentioned* sharing can leak Personally Identifiable Information (PII) or location cues. High‑profile incidents (e.g., inadvertent exposure of conversation metadata via AI platform bugs, third‑party plugin mishandling of chat content, and prompt logging leading to accidental sensitive data retention) have underscored three converging risks:

- Accidental leakage: Users post screenshots (bank transfers, boarding passes, school IDs) without realizing embedded micro‑identifiers (names, building numbers, GPS EXIF remnants).
- Inferential privacy attacks: Seemingly benign visual or textual fragments (street signs, license plates, store fronts, neighborhood landmarks) enable geo‑location or identity linkage.
- Inconsistent manual redaction: Users who *try* to redact rely on ad‑hoc blurring with image editors, often missing secondary occurrences (e.g., names in notification banners, mirrored reflections, chat previews).

As AI foundation models get better at entity linking and multimodal correlation, the risk surface expands: what was once “obscure” becomes trivially recoverable by automated inference. Thus, privacy resilience must move *upstream*, *before* content leaves the user’s device or is transmitted to cloud AI services.

## 2. Specific Problem We Tackle

We target potential **privacy leakage in social media posting and commenting workflows (e.g., TikTok)**:

- Users composing a text caption or comment may include phone numbers, addresses, dates of birth, banking hints, or contact handles.
- Users uploading images or video frames may unknowingly include: faces of bystanders, chat app screenshots, notification overlays, street signs, license plates, or identifiable interior artifacts.
- Existing manual workflows (open an editor → blur → hope nothing is missed) are slow, error‑prone, and discourage consistent anonymization.

**Goal:** Provide a *fast, precise, multimodal pre‑publication privacy sentinel* that detects and masks sensitive regions or tokens with minimal friction.

## 3. Our Solution – Sentinel

Sentinel is a **multimodal privacy preprocessing layer** combining:

1. **Precision Detection**  
   - Text: High‑granularity, 58‑class PII tagging (names, addresses, financial IDs, government IDs, technical identifiers).  
   - Images / Video: Fast face detection + batch anonymization; extensible pipeline for higher‑level sensitive object / text (future module).
2. **Automated + Assisted Masking**  
   - For privacy‑aware users: A *selective*, GUI-driven workflow enabling interactive face inclusion/exclusion and anonymization mode choice (blur, pixelate, emoji overlay, synthetic replacement).
   - For privacy‑unaware users: *Default safe mode* auto-masks all detected sensitive entities/faces instantly.
3. **Speed & Coverage**  
   - Lightweight inference paths for common operations (on-device or edge-capable models).
   - Reduced risk of “missed spot” through token‑level PII enumeration and per‑face region proposals.
4. **Consistency & Trust**  
   - Deterministic anonymization transforms (with optional randomness for synthetic replacements).
   - Confidence score thresholds tunable to user risk tolerance.

**User Experience:** “Drag or select content → Sentinel preprocesses in split seconds → Preview (optional) → Safe output is posted.”

## 4. Inner Working Mechanism (Architecture Walkthrough)

### 4.1 Overview Pipeline

```
[ User Input ]
   ├── Text (caption/comment)
   │     → Tokenization → PII Model Inference → Entity Spans → Masking / Redaction Rules → Sanitized Text
   ├── Image / Video
   │     ├── Fast Face Mode
   │     │     → Frame Acquisition
   │     │     → YOLOv12l-face Detection
   │     │     → Face Region Proposals
   │     │     → Mode Selection (blur | pixelate | emoji | synthetic swap)
   │     │     → Post-processing (bystander logic, compositing)
   │     │     → Safe Media
   │     └── Deep Mode (Advanced Visual Sensitive Detector)
   │           → Frame Acquisition
   │           → Multi-Stage Analysis:
   │                1. Face Detection (YOLOv12l-face)
   │                2. OCR Extraction (screen text, chat bubbles, signage, ID text)
   │                3. Contextual Object / Scene Cues (street signs, license plates, UI panels)
   │                4. Geo/Identity Risk Scoring (toponyms, number patterns, branded/logotype regions)
   │           → Region Fusion & Deduplication (merge overlapping face+text+object boxes)
   │           → Classification of Region Type (FACE | TEXT-PII | LOCATION-CUE | ID-DOCUMENT | SCREENSHOT-UI)
   │           → Policy-Based Transform (mask | blur | pixelate | synthetic replace | remove)
   │           → Confidence-Aware Review (optional user override in GUI)
   │           → Safe Media
   └── Audio (planned)
         → Speech-to-Text → PII Tagging → Segment Redaction / Bleep → Re-aligned Transcript (future module)
```
**Deep Mode Notes**
- OCR output is immediately piped through the same PII text classifier for high-resolution token-level redaction within detected text regions.
- Object / scene cue detection augments OCR to catch non-textual location hints (e.g., distinctive storefront signage, license plates).
- Region Fusion ensures a single masking pass even when multiple detectors flag overlapping areas (e.g., a face inside a chat screenshot).
- Risk scoring prioritizes masking order; higher-risk regions (e.g., government ID text) are applied first to avoid partial exposure during preview.

### 4.2 Text PII Detection Subsystem

Referenced components (see `pii-text-detector(train)/`):
- Multiple architectures trained / benchmarked:  
  - Ettin 1B (custom encoder; top F1 macro ≈ 0.9883)  
  - Ettin 400M  
  - DeBERTa v3 Base  
  - ModernBERT Base  

Key steps:
1. **Preprocessing**: Input string normalized (Unicode NFC, whitespace compaction).
2. **Tokenization**: Model-specific (e.g., SentencePiece / WordPiece) producing token IDs + attention masks.
3. **Inference**: Sequence labeling (NER-style) predicting per-token tags among 58 PII categories.
4. **Span Reconstruction**: Merge contiguous tokens with same label; handle subword boundaries.
5. **Confidence Filtering**: Default threshold (e.g., 0.5). Exposed for user risk tuning.
6. **Redaction Strategies** (configurable):
   - Full removal: Replace with `[REDACTED_TYPE]`.
   - Partial masking: E.g., email `j***@domain.com`.
   - Hashing: Deterministic salted hash for pseudonymization while preserving referentiality.
7. **Audit Map Output**: Provide JSON manifest: `{ "type": "EMAIL", "text": "...", "confidence": 0.97, "start": 34, "end": 49 }` enabling traceability.

### 4.3 Image Face Anonymization Subsystem

Referenced code: `face_anonymizer.py`, `face_anonymizer_main.py`, `face_anonymizer_gui_modern.py`.

Core components:
1. **Detection**: YOLOv12l-face model run image producing bounding boxes + confidences.
2. **Selection Logic**:
   - Default: All faces except (optionally) primary subject (largest bounding box) to protect bystanders while preserving creator identity.
   - GUI: Click to toggle per-face anonymization (`face_anonymizer_gui_modern.py` handles interactive event mapping).
3. **Anonymization Modes**:
   - Blur: Adaptive Gaussian kernel scaled to face size (ensures minimum obfuscation entropy).
   - Pixelate: Downscale–upscale blocks; interpolation selectable (Nearest / Linear / Cubic).
   - Emoji Overlay: Alpha-composited PNG from `/emoji/` assets; auto-resized to bounding box with centering offsets.
   - Synthetic Replacement: InsightFace InSwapper (`inswapper_128.onnx`) swaps random or uniform synthetic face—preserves natural lighting & pose.
4. **Performance Optimizations**:
   - Model warm start (load once).
   - (Optional) Confidence > threshold short-circuits low-prob detections.
   - Vectorized pixelation / blur kernels.
5. **Safety Envelope**:
   - Expand bounding box with margin factor to avoid hairline leakage.
   - Resize operations maintain aspect ratio; fallback cropping if bounds exceed canvas.

### 4.4 Planned / Extensible Modules (Phase 2)

(Planned, not yet fully implemented in this repository—documenting roadmap transparently.)

1. **Advanced Visual Sensitive Detector**  
   - OCR (e.g., PaddleOCR / Tesseract) + text PII pipeline bridging (feed recognized text through same Ettin-based classifier).  
   - Object detection for location cues: street signs, license plates, store logos (YOLO variant or Grounding DINO).  
   - Saliency scoring for “geo-inference risk” (weighted combination of recognized toponyms + signage count).

2. **Audio Redaction (Future)**  
   - Speech-to-Text (on-device Whisper small / Distil-Whisper) → PII labeling → bleep / replace segments → re-synthesize timestamps.

3. **EXIF & Metadata Sanitizer**  
   - Strip GPS, device model, serial fields prior to upload.

### 4.5 Masking Policy Engine

A rules layer (conceptual) selects transform strategy based on:
- PII Type Risk Tier (e.g., SSN → full removal; AGE → partial)
- User Mode (Paranoid / Balanced / Minimal)
- Contextual Redundancy (repeat exposures collapsed into single placeholder to reduce cognitive clutter)

Policy Example:

| PII Type | Default Mode | Optional Mode |
|----------|--------------|---------------|
| CREDITCARDNUMBER | Full redact | Tokenize (last 4) |
| EMAIL | Partial mask | Hash |
| FIRSTNAME/LASTNAME | Hash | Pseudonym dictionary |
| GPS COORD / ADDRESS | Remove | Coarse generalization (city-level) |

### 4.6 Confidence-Aware UX

- Spans / faces with confidence in “gray zone” (e.g., 0.4–0.6) can surface an interactive checkbox (GUI) flagged “Review”.
- Users may whitelist tokens (e.g., brand names) to refine future sessions (local preferences file; no cloud logging).

### 4.7 Privacy & Security Design Considerations

- On-device inference first; no raw media leaves device for detection.
- Ephemeral buffers cleared after export.
- Optional “memory off” mode: disables caching of detection results.
- Deterministic synthetic face seed can be reset per session to avoid pattern correlation.

### 4.8 Multimodality Focus

Current: Text + Image + (Video frames treated as image sequences).  
Roadmap: Audio (speech), Screen-captured sequences (temporal OCR), Cross-modal correlation (e.g., name extracted from text triggers face anonymization priority list if caption references a person).

### 4.9 Evaluation & Metrics

| Dimension | Metric | Current Approach |
|-----------|--------|------------------|
| Text PII | Macro F1 | Reported in model README (Ettin 1B ~0.9883) |
| Image Faces | Detection Recall | Empirical YOLO face test vs. sample set |
| Anonymization Strength | Re-identification drop (%) | (Planned) Face embedding cosine similarity pre/post swap |
| Latency | Avg ms / (image or 1080p frame) | Bench after warm load |
| False Positives | % benign tokens masked | Threshold calibration set |

### 4.10 Extensibility

Plugin hooks:
- `pre_text_mask` / `post_text_mask`
- `pre_frame_process` / `post_frame_process`
- Additional anonymizers (e.g., stylization GAN) implementing a simple interface:
  ```
  def apply(region: np.ndarray, config: Dict) -> np.ndarray
  ```

## 5. Advantages of Sentinel

1. **Lightweight Yet Capable (≤ ~3B Parameter Upper Bound Path)**  
   - Our deployed ensemble favors *practical* model sizes (Ettin 1B + optional variants) and is architected to support an extended multimodal core up to ~3B parameters while still remaining laptop‑runnable.  
   - With mixed-precision (FP16 / INT8 / 4-bit quantization) a modern consumer GPU (e.g., RTX 3050 / Apple M-series unified memory) or even CPU-only fallback can operate in acceptable latency envelopes.

2. **Fully Local Processing – Zero Secondary Exposure Risk**  
   - 100% on-device inference: no outbound API calls for detection or anonymization.  
   - Eliminates “secondary leakage” / *downstream retention vectors* (i.e., risk that a third-party model provider logs prompts or retains media).  
   - Reduces data-in-use attack surface and aligns with zero-trust / data minimization principles.

3. **Rich Anonymization & Operation Modes**  
   - Multiple *anonymization transforms* (blur, pixelate, emoji overlay, synthetic face replacement) plus *operational modes* (Auto, Assisted Review, Batch).  
   - Policy engine enables per-type strategies (mask vs. hash vs. pseudonymize).

4. **High Throughput / Low Latency**  
   - Warm-start pipeline, region-level processing, vectorized ops, and selective model invocation keep processing near real-time for common resolutions (e.g., sub-second for typical images; streaming viability for video).  
   - Text PII detection leverages efficient transformer encoders with high F1 but controlled parameter counts.

5. **Deterministic & Auditable**  
   - Optional manifest export of all redactions (with offsets & confidence) for compliance or peer review.

6. **Extensible Modular Design**  
   - Plugin hooks allow community contributions (e.g., license plate detector, OCR module, dataset curation filters) without modifying core.

7. **User-Centric Safety Defaults**  
   - Conservative auto-mask mode prevents accidental oversharing for users who are unaware of privacy best practices.

## 6. Extended Use Cases Beyond Social Posting

Although we began by focusing on TikTok-style posting & commenting, the applicability is far broader:

- **Email Composition Sanitizer**: Strip accidental signatures (personal phone, address) before sending to broad lists.
- **Prompt Preprocessor for LLMs**: Remove client names, internal project codenames, API keys from prompts before hitting external AI endpoints.
- **Customer Support & CRM Logs**: Automated redaction prior to exporting tickets for analytics / model fine-tuning.
- **Screenshot & Screen Recording Sanitization**: Mask chat pop-ups, notification banners, identifiers in tutorial videos or bug reports.
- **Bug / Issue Reporting Pipelines**: Integrate as a pre-commit hook to redact secrets (tokens, keys) or user data.
- **Open Dataset Curation**: Prepare image / text corpora by anonymizing individuals and sensitive textual spans.
- **HR & Talent Review**: Anonymize candidate CV fragments for bias mitigation workflows.
- **Legal Discovery / e-Disclosure**: Rapid triage redaction of personal fields before external counsel sharing.
- **Telemedicine / Health Notes (Non-diagnostic)**: Strip patient identifiers in exported summaries (aligning with minimal PHI exposure goals).
- **Educational Content Creation**: Blur student faces & names in classroom recording extracts.
- **Enterprise Chat Gatekeeper**: Pre-send filter for internal Slack/Teams messages to prevent unintentional data leakage across channels.
- **Call Center + Meeting Transcripts (Future Audio Module)**: Detect and redact phone numbers, account IDs in transcripts prior to training analytics models.
- **Research Publication Prep**: Anonymize field study media while preserving utility for analysis.

By functioning as a *universal privacy preprocessor*, Sentinel lowers friction to enforce consistent data hygiene across creative, operational, compliance, and analytical workflows.

## 7. Technical Model Details (Placeholder)

(To be expanded)
- YOLOv12l-face: architecture summary (backbone, input size, anchor-free heads).
- Ettin Encoder (1B / 400M / planned up to 3B multimodal) parameter breakdown: embedding dims, attention depth, positional encoding.
- DeBERTa / ModernBERT comparative parameterization.
- Inference optimization: quantization strategy, ONNX Runtime graph fusions, potential TensorRT plans.
- Memory & throughput benchmarks (CPU vs. GPU vs. Apple Silicon).
- Synthetic face swap pipeline: alignment → embedding extraction → conditional generation.

## 8. Additional Notes & Ethical Guardrails

- No re-identification modules will be added (explicit non-goal).
- Redaction logs stored locally only if user opts in for audit compliance (e.g., enterprise edition).
- Transparent disclaimers: “Automated masking cannot guarantee absolute anonymity; users retain final responsibility.”
- Open source licensing encourages community audits for privacy integrity.
- Potential integration with platform publishing SDKs so Sentinel acts as a *pre-flight privacy gate*.

## 9. References

- (Model Performance) Internal benchmarking documented in `pii-text-detector(train)/README.md`.
- (Face Detection) YOLO face model release: https://github.com/YapaLab/yolo-face
- (Face Swapping Backbone) InsightFace InSwapper project & model weights.
- (PII Categories Guidance) NIST Privacy Framework; general data taxonomy references.
- (Privacy Risk Escalation) Public discussions on inadvertent AI data exposure & logging practices.
- (OCR + Sensitive Text Risk) Research on screenshot and document leakage in social media ecosystems.
- (Geo-Location Inference) Studies on image-based location prediction and scene text correlation.

## 10. Summary

Sentinel operationalizes proactive, multimodal privacy defense for social media creators and a wide range of adjacent workflows: **detect → decide → anonymize**, compressing a traditionally manual, error-prone chain into a subsecond, user-friendly workflow. With strong PII recall, flexible masking strategies, a lean on-device footprint (≤ ~3B parameters for extended multimodality), and an extensible architecture, it forms a practical blueprint toward “privacy-by-default” content creation and data handling.

---

*Prepared for the TechJam: Privacy Meets AI challenge. Sections marked as planned are transparently disclosed future roadmap items; the current repository contains the core face and text anonymization engines.*
