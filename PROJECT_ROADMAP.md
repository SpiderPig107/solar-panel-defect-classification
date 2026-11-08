# Solar Panel Defect Classifier: Project Roadmap & Governance

**Project Name**: Solar Panel Images Clean and Faulty Classification  
**Target Dataset**: ~869 images across 6 classes (`Clean`, `Dusty`, `Bird-drop`, `Electrical-damage`, `Physical-Damage`, `Snow-Covered`)  
**Architecture**: Pretrained Backbone (EfficientNetB0 / ResNet50) + Transfer Learning Head  
**Primary Workflow**: Local prototyping in `draft.ipynb` -> Final training on Kaggle GPUs  

---

## Master Checklist

- [x] **Phase 1: Exploratory Data Analysis (EDA)**
  - [x] 1.1 Verify dataset location and directory integrity.
  - [x] 1.2 Check for corrupt / unreadable image files.
  - [x] 1.3 Count images per class and plot class distribution (detect imbalance).
  - [x] 1.4 Generate visual gallery (sample photo from each of the 6 classes).

- [x] **Phase 2: Data Pipeline & Stratified Splitting**
  - [x] 2.1 Implement stratified 80:20 Train/Validation split (`seed=42`).
  - [x] 2.2 Standardize image resolution to `(224, 224)`.
  - [x] 2.3 Set batch size (`16`) and configure `tf.data` prefetching.

- [x] **Phase 3: Domain-Specific Data Augmentation**
  - [x] 3.1 Configure horizontal and vertical flips (`RandomFlip`).
  - [x] 3.2 Configure small rotation (`RandomRotation(0.15)`) and contrast (`RandomContrast(0.2)`).
  - [x] 3.3 Apply augmentation strictly to training data only.

- [x] **Phase 4: Baseline Transfer Learning Architecture**
  - [x] 4.1 Instantiate pretrained backbone with ImageNet weights (`include_top=False`).
  - [x] 4.2 Freeze backbone weights (`trainable = False`).
  - [x] 4.3 Attach classification head: `GlobalAveragePooling2D`, `Dropout(0.2)`, `Dense(6, activation='softmax')`.

- [x] **Phase 5: Baseline Training & Diagnostics**
  - [x] 5.1 Compile with `Adam(learning_rate=1e-3)`, `sparse_categorical_crossentropy`, `metrics=['accuracy']`.
  - [x] 5.2 Configure `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`.
  - [x] 5.3 Train model and plot loss/accuracy curves.

- [x] **Phase 6: Precision Fine-Tuning**
  - [x] 6.1 Unfreeze top 10 backbone layers (keep BatchNormalization frozen).
  - [x] 6.2 Recompile with small learning rate (`Adam(learning_rate=1e-5)`).
  - [x] 6.3 Train for 10 additional epochs and record performance gain (`val_loss: 0.6084`).

- [ ] **Phase 7: Evaluation, Error Analysis & Export**
  - [x] 7.1 Single-pass evaluation & Classification Report (Precision, Recall, F1-Score across all 6 classes).
  - [x] 7.2 Confusion Matrix visualization and interpretation.
  - [ ] 7.3 Visual inspection of misclassifications.
  - [ ] 7.4 Final Model Recommendation & Production Export (`solar_defect_efficientnetb0.keras`).
  - [x] 7.5 *(Skipped by User Decision)* Grad-CAM visual attention heatmaps.

---

## Active Phase Log

* **Current Active Phase**: Phase 7 (Evaluation, Error Analysis & Export)
* **Status**: Phases 1, 2, 3, 4, 5, and 6 are 100% complete and signed off. Phase 6 fine-tuning completed with lowest project loss (val_loss: 0.6084 at Epoch 30). Active work advancing to Phase 7 evaluation and export.
* **Key Architecture Decisions Logged**:
  * **Option B Confirmed**: Preserved raw `Snow-Covered` benchmark dataset as-is (including surrounding roof snow). Will audit context shortcut learning in Phase 7 via Grad-CAM (Item 7.5).
  * **Aspect Ratio Ingestion**: 41 images ($4.6\%$) outside $[0.5, 2.0]$ confirmed safe; pipeline configured with `crop_to_aspect_ratio=False` to prevent discarding edge defects.
  * **Color Normalization**: 19 RGBA images confirmed; pipeline configured with `channels=3` to defensively strip alpha channels into clean 3-channel tensors.
  * **Stratified 80:20 Split**: 708 train / 177 validation images verified via `train_test_split(stratify=all_labels, seed=42)`. Exactly 20% validation per class achieved.
  * **Domain-Specific Augmentation**: Integrated into model via `RandomFlip("horizontal_and_vertical")`, `RandomRotation(0.15)`, and `RandomContrast(0.2)`. Deactivates automatically during validation/inference (`training=False`).
  * **Architecture Confirmed**: `EfficientNetB0` backbone + `GlobalAveragePooling2D` + `Dropout(0.2)` + `Dense(6, 'softmax')` head.
  * **Class Imbalance Handling**: Applied balanced class weights during `model.fit()` in Phase 5. Produced 90.5% recall on electrical damage and 83.3% precision on physical damage.
  * **Phase 5 Baseline Results**: 80.8% Accuracy, 0.812 Macro F1 across 177 validation samples. Baseline plateau reached at Epoch 18 (val_loss: 0.6285).
  * **Phase 6 Fine-Tuning Results**: Top 10 layers unfrozen with `BatchNormalization` locked. Compiled at `lr=1e-5`. Achieved record low `val_loss = 0.6084`, $88.26\%$ train accuracy, $83.05\%$ val accuracy, tightly controlled $5.21\%$ generalization gap.

