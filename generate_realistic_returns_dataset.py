import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# Classes
CLASSES = ['restock', 'refurbish', 'recycle']

# Example phrases for each class (with overlap and ambiguity)
REASONS = {
    'restock': [
        'Wrong size ordered', 'Changed mind', 'No longer needed', 'Duplicate order',
        'Received as gift', 'Ordered wrong quantity', 'Better price found',
        'Family member disapproval', 'Fit not as expected', 'Color not as shown',
        'Material feel different', 'Style doesn\'t match', 'Occasion cancelled',
        'Size chart was incorrect', 'Product arrived late', 'Budget constraints',
        'Different style preferred', 'Seasonal preference change', 'Gift return',
        'Did not like the product', 'Unwanted purchase'
    ],
    'refurbish': [
        'Minor cosmetic damage', 'Packaging slightly damaged', 'Small scratch',
        'Box opened but item unused', 'Minor dent', 'Slight color fading',
        'Small tear', 'Minor button issue', 'Loose thread', 'Small stain',
        'Slight wear on edges', 'Minor functionality issue', 'Cosmetic imperfection',
        'Adhesive residue', 'Small chip', 'Minor alignment issue',
        'Slight discoloration', 'Small fabric pull', 'Minor scuff mark',
        'Light usage wear', 'Small puncture', 'Minor zipper issue',
        'Slight stretching', 'Small hole', 'Minor warping',
        # Overlap with recycle
        'Heavy wear', 'Major scratch', 'Significant fading', 'Noticeable dent',
        'Obvious repair needed', 'Obvious usage marks'
    ],
    'recycle': [
        'Completely broken', 'Major malfunction', 'Severe water damage',
        'Fire damage', 'Multiple components failed', 'Irreparable damage',
        'Major structural damage', 'Extensive wear and tear', 'Beyond repair',
        'Hazardous material exposure', 'Severe corrosion', 'Major impact damage',
        'Electrical short circuit', 'Overheating damage', 'Chemical damage',
        'Extreme weather damage', 'Manufacturing defect', 'Battery leak',
        'Cracked screen', 'Motor failure', 'Torn beyond repair', 'Severe fading',
        'Permanent staining', 'Fabric completely deteriorated', 'Structural failure',
        'Safety hazard',
        # Overlap with refurbish
        'Heavy wear', 'Major scratch', 'Significant fading', 'Noticeable dent',
        'Obvious repair needed', 'Obvious usage marks'
    ]
}

# Inspector notes templates (with ambiguity and noise)
INSPECTOR_TEMPLATES = [
    'Inspected: {reason}. Needs further review.',
    'Customer reported: {reason}.',
    'Observed: {reason}.',
    'Possible issue: {reason}.',
    'Returned due to: {reason}.',
    'Notes: {reason}.',
    'Initial check: {reason}.',
    'Visual inspection: {reason}.',
    'Flagged: {reason}.',
    'Assessment: {reason}.',
    'Possible {class_label} case. {reason}.',
    'Unclear if {class_label} or other. {reason}.',
    'May require {class_label}. {reason}.',
    'Uncertain: {reason}.',
    'Reported: {reason}.',
    'See attached: {reason}.',
    'Customer notes: {reason}.',
    'Potential {class_label} situation. {reason}.',
    'Further testing needed: {reason}.',
    'Possible overlap with other categories. {reason}.'
]

# Generate synthetic dataset
def generate_realistic_returns_dataset(n_samples=20000, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    data = []
    for _ in range(n_samples):
        class_label = random.choices(CLASSES, weights=[0.4, 0.35, 0.25])[0]
        # Sometimes pick a reason from another class for ambiguity
        if random.random() < 0.15:
            other_class = random.choice([c for c in CLASSES if c != class_label])
            reason = random.choice(REASONS[other_class])
        else:
            reason = random.choice(REASONS[class_label])
        # Add noise/variation
        if random.random() < 0.1:
            reason = reason.lower()
        if random.random() < 0.05:
            reason = reason + ' (customer unclear)'
        inspector_note = random.choice(INSPECTOR_TEMPLATES).format(reason=reason, class_label=class_label)
        # Add typos or synonyms
        if random.random() < 0.05:
            inspector_note = inspector_note.replace('inspection', 'inspction')
        if random.random() < 0.05:
            inspector_note = inspector_note.replace('review', 'revue')
        data.append({
            'inspector_notes': inspector_note,
            'return_reason': reason,
            'classification': class_label
        })
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    df = generate_realistic_returns_dataset()
    df.to_csv('data/product_classification_dataset_realistic.csv', index=False)
    print("Generated data/product_classification_dataset_realistic.csv with realistic ambiguity and noise.")
