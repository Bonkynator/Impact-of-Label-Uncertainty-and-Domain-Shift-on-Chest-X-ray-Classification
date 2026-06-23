CheXpert Data Drift Study
Research Question: How does uncertain label handling affect model robustness to distribution shift in medical imaging?

This codebase implements a systematic comparison of three uncertainty handling strategies for training chest X-ray classification models, then evaluates their performance under distribution shift.

Project overview
Experimental design
Train three DenseNet-121 models on CheXpert dataset, each with different uncertain label strategy:

U-Zeros: Uncertain labels -> 0 (negative)
U-Ones: Uncertain labels -> 1 (positive)
U-Half: Uncertain labels -> 0.5 (soft label)
Evaluate each model on:

CheXpert validation set (in-distribution baseline)
ChestX-ray14 dataset (out-of-distribution / distribution shift)
Analyze which strategy is most robust to distribution shift

Key contribution
This work quantifies how different approaches to handling label uncertainty during training affect a model's ability to generalize across different data distributions—a critical consideration for deploying medical AI systems across multiple clinical sites.

Project structure
chexpert_drift_study/
├── config.py              # All hyperparameters and settings
├── main.py                # Main execution script
├── requirements.txt       # Python dependencies
├── utils/
│   ├── dataset.py        # Data loading and preprocessing
│   ├── model.py          # DenseNet architecture
│   ├── train.py          # Training loop
│   └── evaluate.py       # Evaluation and metrics
├── data/                 # Download datasets here
├── models/               # Trained model checkpoints
├── results/              # Evaluation results and plots
└── notebooks/            # Jupyter notebooks for analysis
Setup instructions
1. Install dependencies
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
2. Download datasets
CheXpert Dataset
Visit: https://www.kaggle.com/datasets/ashery/chexpert/
Download CheXpert-v1.0-small
Extract to data/chexpert
Expected structure:

data/chexpert/
├── train/
│   ├── patient64541/
│   │   ├── study1/
│   │   │   └── view1_frontal.jpg
│   │   └── study2/
│   └── ...
├── valid/
├── train.csv
└── valid.csv
ChestX-ray14 Dataset
Visit: https://www.kaggle.com/datasets/nih-chest-xrays/data
Download NIH Chest X-rays
Extract all to data/NIH/
Expected structure:

data/NIH/
├── images/
│   ├── 00000001_000.png
│   ├── 00000001_001.png
│   └── ...
└── Data_Entry_2017.csv
3. Update configuration
Edit config.py and update these paths (change accordingly):

CHEXPERT_DIR = "/data/chexpert" 
CHESTXRAY14_DIR = "/data/NIH"
Running the pipeline
Option 1: Train and evaluate everything
python main.py --mode all
This will:

Train all 3 models (U-Zeros, U-Ones, U-Half)
Evaluate each on CheXpert validation
Evaluate each on ChestX-ray14
Generate comparison tables and plots
Expected time: 8-12 hours per model on a modern GPU (~36 hours total)

Option 2: train Only
python main.py --mode train
Option 3: Evaluate only (after training)
python main.py --mode eval
Understanding the code
How uncertainty handling works
In utils/dataset.py, the CheXpertDataset class handles uncertain labels:

# CheXpert labels are encoded as:
# - 1.0: Positive (disease present)
# - 0.0: Negative (disease absent)
# - -1.0: Uncertain (radiologist unsure)
# - NaN: Not mentioned

# Our code replaces -1.0 with the strategy value:
for label in self.labels:
    uncertain_mask = self.df[label] == -1.0
    self.df.loc[uncertain_mask, label] = self.uncertainty_strategy
Training loop
In utils/train.py, the main training function:

Forward pass: Images -> Model -> Predictions
Compute loss: Binary cross-entropy (allows soft labels)
Backward pass: Compute gradients
Update weights: Optimizer adjusts parameters
Validate: Check performance on held out data
Early stopping: Stop if no improvement for N epochs
Evaluation metrics
The code computes:

AUC-ROC: Area under receiver operating characteristic curve (main metric)
Sensitivity: True positive rate
Specificity: True negative rate
Performance drop: In dist AUC - Out dist AUC
Understanding the research
Why this matters
Medical AI models often fail when deployed to new hospitals because:

Different equipment (GE vs Siemens scanners)
Different patient demographics
Different imaging protocols
Contribution: Showing that how we handle uncertain labels during training affects this robustness.

The hypothesis
H1: U-Zeros will underperform (treats uncertain positives as negative -> mislabeled data)

H2: U-Ones might overfit (treats weak signals as strong positives)

H3: U-Half will be most robust (soft labels preserve uncertainty -> better calibration)

Expected outcome
U-Half strategy should show:

Similar in-distribution performance to other strategies
Better out-of-distribution performance
Smallest performance drop
This proves: Soft labels help models generalize better

Current status (May 2026)
I am currently navigating hardware constraints to execute the full 36 hour training benchmarks across the 60GB dataset. The repository will be updated with final AUC-ROC metrics upon completion.
