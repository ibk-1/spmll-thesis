"""
Automatic Rule Mining from Label Co-occurrence Statistics
==========================================================

This module mines implication and exclusion rules from training data
using statistical thresholds.
"""

import numpy as np
import json
from pathlib import Path
from collections import defaultdict


class RuleMiner:
    """
    Mine logical rules from multi-label data.
    
    Implication (A → B): P(B|A) is high
    Exclusion (A ⊥ B): P(A ∧ B) is low AND both classes appear frequently
    """
    
    def __init__(self, labels, class_names, min_support=10):
        """
        Args:
            labels: numpy array of shape (N, C) with binary labels
            class_names: dict mapping class index to name
            min_support: minimum number of occurrences for a class to be considered
        """
        self.labels = labels
        self.class_names = class_names
        self.num_samples, self.num_classes = labels.shape
        self.min_support = min_support
        
        # Compute statistics
        self._compute_statistics()
    
    def _compute_statistics(self):
        """Compute co-occurrence and conditional probability statistics."""
        print("Computing label statistics...")
        
        # Class frequencies: P(A)
        self.class_freq = self.labels.sum(axis=0)  # (C,)
        self.class_prob = self.class_freq / self.num_samples
        
        # Co-occurrence: P(A ∧ B)
        self.cooccurrence = self.labels.T @ self.labels  # (C, C)
        self.cooccurrence_prob = self.cooccurrence / self.num_samples
        
        # Conditional probabilities: P(B|A) = P(A ∧ B) / P(A)
        self.conditional_prob = np.zeros((self.num_classes, self.num_classes))
        for a in range(self.num_classes):
            if self.class_freq[a] > 0:
                self.conditional_prob[a, :] = self.cooccurrence[a, :] / self.class_freq[a]
        
        # Pointwise Mutual Information: PMI(A,B) = log(P(A,B) / (P(A)*P(B)))
        self.pmi = np.zeros((self.num_classes, self.num_classes))
        for a in range(self.num_classes):
            for b in range(self.num_classes):
                if a != b and self.class_freq[a] > 0 and self.class_freq[b] > 0:
                    p_ab = self.cooccurrence_prob[a, b]
                    p_a = self.class_prob[a]
                    p_b = self.class_prob[b]
                    if p_ab > 0:
                        self.pmi[a, b] = np.log(p_ab / (p_a * p_b))
        
        print(f"✓ Statistics computed")
        print(f"  Dataset size: {self.num_samples} images")
        print(f"  Number of classes: {self.num_classes}")
        print(f"  Average labels per image: {self.labels.sum() / self.num_samples:.2f}")
    
    def mine_implication_rules(self, 
                               confidence_threshold=0.7,
                               lift_threshold=1.5,
                               min_support=None):
        """
        Mine implication rules A → B.
        
        A good implication satisfies:
        1. P(B|A) ≥ confidence_threshold (high conditional probability)
        2. P(B|A) / P(B) ≥ lift_threshold (B appears MORE often with A)
        3. freq(A) ≥ min_support (A appears frequently enough)
        
        Args:
            confidence_threshold: minimum P(B|A) to consider (0.7 = 70%)
            lift_threshold: minimum lift = P(B|A) / P(B)
            min_support: minimum frequency of class A
            
        Returns:
            List of (A, B, confidence, lift) tuples
        """
        if min_support is None:
            min_support = self.min_support
        
        rules = []
        
        for a in range(self.num_classes):
            if self.class_freq[a] < min_support:
                continue  # Skip rare classes
            
            for b in range(self.num_classes):
                if a == b:
                    continue
                if self.class_freq[b] < min_support:
                    continue
                
                confidence = self.conditional_prob[a, b]
                
                if confidence < confidence_threshold:
                    continue
                
                # Compute lift: how much more likely is B when A is present?
                lift = confidence / (self.class_prob[b] + 1e-10)
                
                if lift < lift_threshold:
                    continue
                
                rules.append((a, b, confidence, lift))
        
        # Sort by confidence * lift (combined score)
        rules.sort(key=lambda x: x[2] * x[3], reverse=True)
        
        return rules
    
    def mine_exclusion_rules(self,
                            max_cooccurrence_prob=0.05,
                            min_individual_prob=0.05):
        """
        Mine exclusion rules A ⊥ B.
        
        A good exclusion satisfies:
        1. P(A ∧ B) ≤ max_cooccurrence_prob (rarely appear together)
        2. P(A) ≥ min_individual_prob AND P(B) ≥ min_individual_prob (both common)
        
        This finds classes that are individually common but rarely co-occur.
        
        Args:
            max_cooccurrence_prob: maximum P(A ∧ B) to consider
            min_individual_prob: minimum P(A) and P(B)
            
        Returns:
            List of (A, B, P(A∧B), P(A), P(B)) tuples
        """
        rules = []
        
        for a in range(self.num_classes):
            if self.class_prob[a] < min_individual_prob:
                continue
            
            for b in range(a + 1, self.num_classes):  # Only upper triangle
                if self.class_prob[b] < min_individual_prob:
                    continue
                
                cooccur_prob = self.cooccurrence_prob[a, b]
                
                if cooccur_prob > max_cooccurrence_prob:
                    continue
                
                # Compute expected co-occurrence under independence
                expected = self.class_prob[a] * self.class_prob[b]
                
                # Only keep if actual << expected (negative association)
                if cooccur_prob < 0.5 * expected:
                    rules.append((a, b, cooccur_prob, self.class_prob[a], self.class_prob[b]))
        
        # Sort by cooccurrence probability (lowest first = strongest exclusion)
        rules.sort(key=lambda x: x[2])
        
        return rules
    
    def print_rules(self, implication_rules, exclusion_rules, top_k=20):
        """Pretty print mined rules."""
        print("\n" + "="*80)
        print("MINED IMPLICATION RULES (A → B)")
        print("="*80)
        print(f"Found {len(implication_rules)} rules. Showing top {min(top_k, len(implication_rules))}:\n")
        
        for i, (a, b, conf, lift) in enumerate(implication_rules[:top_k], 1):
            a_name = self.class_names[a]
            b_name = self.class_names[b]
            a_freq = int(self.class_freq[a])
            b_freq = int(self.class_freq[b])
            cooccur = int(self.cooccurrence[a, b])
            
            print(f"{i:2d}. {a_name:20s} → {b_name:20s}")
            print(f"    Confidence: {conf:.3f} ({cooccur}/{a_freq})")
            print(f"    Lift: {lift:.2f}x")
            print(f"    Interpretation: When {a_name} appears, {b_name} appears {conf*100:.1f}% of the time")
            print()
        
        print("\n" + "="*80)
        print("MINED EXCLUSION RULES (A ⊥ B)")
        print("="*80)
        print(f"Found {len(exclusion_rules)} rules. Showing top {min(top_k, len(exclusion_rules))}:\n")
        
        for i, (a, b, cooccur_prob, prob_a, prob_b) in enumerate(exclusion_rules[:top_k], 1):
            a_name = self.class_names[a]
            b_name = self.class_names[b]
            a_freq = int(self.class_freq[a])
            b_freq = int(self.class_freq[b])
            cooccur_count = int(self.cooccurrence[a, b])
            
            print(f"{i:2d}. {a_name:20s} ⊥ {b_name:20s}")
            print(f"    Co-occurrence: {cooccur_prob:.4f} ({cooccur_count}/{self.num_samples})")
            print(f"    Individual: P({a_name})={prob_a:.3f}, P({b_name})={prob_b:.3f}")
            print(f"    Interpretation: {a_name} and {b_name} rarely appear together")
            print()
    
    def export_rules(self, implication_rules, exclusion_rules, output_file):
        """Export rules to JSON file."""
        data = {
            'dataset_stats': {
                'num_samples': int(self.num_samples),
                'num_classes': int(self.num_classes),
                'avg_labels_per_image': float(self.labels.sum() / self.num_samples),
                'class_frequencies': {self.class_names[i]: int(self.class_freq[i]) 
                                     for i in range(self.num_classes)}
            },
            'implication_rules': [
                {
                    'rule_id': i,
                    'from_class': int(a),
                    'from_name': self.class_names[a],
                    'to_class': int(b),
                    'to_name': self.class_names[b],
                    'confidence': float(conf),
                    'lift': float(lift),
                }
                for i, (a, b, conf, lift) in enumerate(implication_rules)
            ],
            'exclusion_rules': [
                {
                    'rule_id': i,
                    'class_a': int(a),
                    'name_a': self.class_names[a],
                    'class_b': int(b),
                    'name_b': self.class_names[b],
                    'cooccurrence_prob': float(cooccur_prob),
                    'prob_a': float(prob_a),
                    'prob_b': float(prob_b),
                }
                for i, (a, b, cooccur_prob, prob_a, prob_b) in enumerate(exclusion_rules)
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✓ Rules exported to {output_file}")
    
    def get_rule_pairs(self, implication_rules, exclusion_rules):
        """Get rule pairs in format for training."""
        impl_pairs = [(int(a), int(b)) for a, b, _, _ in implication_rules]
        excl_pairs = [(int(a), int(b)) for a, b, _, _, _ in exclusion_rules]
        return impl_pairs, excl_pairs


def mine_rules_for_dataset(dataset_name, labels_file, class_names,
                           impl_confidence=0.7, impl_lift=1.5,
                           excl_max_cooccur=0.05, excl_min_prob=0.05,
                           top_k_impl=50, top_k_excl=30):
    """
    Complete rule mining pipeline for a dataset.
    
    Args:
        dataset_name: 'pascal' or 'coco'
        labels_file: path to .npy file with labels (N, C)
        class_names: dict mapping index to name
        impl_confidence: confidence threshold for implications
        impl_lift: lift threshold for implications
        excl_max_cooccur: max co-occurrence for exclusions
        excl_min_prob: min individual probability for exclusions
        top_k_impl: keep top K implication rules
        top_k_excl: keep top K exclusion rules
        
    Returns:
        impl_pairs, excl_pairs (lists of tuples)
    """
    print(f"\n{'='*80}")
    print(f"MINING RULES FOR {dataset_name.upper()}")
    print(f"{'='*80}\n")
    
    # Load labels
    print(f"Loading labels from {labels_file}...")
    labels = np.load(labels_file)
    print(f"✓ Loaded labels: shape {labels.shape}")
    
    # Initialize miner
    miner = RuleMiner(labels, class_names, min_support=10)
    
    # Mine implication rules
    print(f"\nMining implication rules (confidence ≥ {impl_confidence}, lift ≥ {impl_lift})...")
    impl_rules = miner.mine_implication_rules(
        confidence_threshold=impl_confidence,
        lift_threshold=impl_lift
    )
    impl_rules = impl_rules[:top_k_impl]  # Keep top K
    print(f"✓ Found {len(impl_rules)} implication rules")
    
    # Mine exclusion rules
    print(f"\nMining exclusion rules (co-occur ≤ {excl_max_cooccur}, P(A),P(B) ≥ {excl_min_prob})...")
    excl_rules = miner.mine_exclusion_rules(
        max_cooccurrence_prob=excl_max_cooccur,
        min_individual_prob=excl_min_prob
    )
    excl_rules = excl_rules[:top_k_excl]  # Keep top K
    print(f"✓ Found {len(excl_rules)} exclusion rules")
    
    # Print rules
    miner.print_rules(impl_rules, excl_rules, top_k=15)
    
    # Export to JSON
    output_file = f"{dataset_name}_mined_rules.json"
    miner.export_rules(impl_rules, excl_rules, output_file)
    
    # Get pairs for training
    impl_pairs, excl_pairs = miner.get_rule_pairs(impl_rules, excl_rules)
    
    return impl_pairs, excl_pairs, miner


# ============================================================================
# CLASS DEFINITIONS
# ============================================================================

# Class names for PASCAL VOC dataset
PASCAL_CLASSES = {
    0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle',
    5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow',
    10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person',
    15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor', 
    20: 'vehicle', 21: 'animal', 22: 'indoor'
}

# Class names for COCO dataset
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep',
    19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe',
    24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
    29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
    34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard',
    38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup',
    42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',
    47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
    52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
    57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
    66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
    75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush',
    80: 'accessory',
    81: 'animal',
    82: 'appliance',
    83: 'electronic',
    84: 'food',
    85: 'furniture',
    86: 'indoor',
    87: 'kitchen',
    88: 'outdoor',
    89: 'sports',
    90: 'vehicle'
}


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Mine logical rules from label data')
    parser.add_argument('--dataset', type=str, required=True, choices=['pascal', 'coco'],
                       help='Dataset name')
    parser.add_argument('--labels-file', type=str, required=True,
                       help='Path to labels .npy file')
    parser.add_argument('--impl-confidence', type=float, default=0.7,
                       help='Confidence threshold for implications')
    parser.add_argument('--impl-lift', type=float, default=1.5,
                       help='Lift threshold for implications')
    parser.add_argument('--excl-max-cooccur', type=float, default=0.05,
                       help='Max co-occurrence for exclusions')
    parser.add_argument('--top-k-impl', type=int, default=50,
                       help='Keep top K implication rules')
    parser.add_argument('--top-k-excl', type=int, default=30,
                       help='Keep top K exclusion rules')
    
    args = parser.parse_args()
    
    # Select class names
    class_names = PASCAL_CLASSES if args.dataset == 'pascal' else COCO_CLASSES
    
    # Mine rules
    impl_pairs, excl_pairs, miner = mine_rules_for_dataset(
        dataset_name=args.dataset,
        labels_file=args.labels_file,
        class_names=class_names,
        impl_confidence=args.impl_confidence,
        impl_lift=args.impl_lift,
        excl_max_cooccur=args.excl_max_cooccur,
        top_k_impl=args.top_k_impl,
        top_k_excl=args.top_k_excl
    )
    
    print(f"\n{'='*80}")
    print("RULE MINING COMPLETE")
    print(f"{'='*80}")
    print(f"Implication rules: {len(impl_pairs)}")
    print(f"Exclusion rules: {len(excl_pairs)}")
    print(f"\nRules saved to {args.dataset}_mined_rules.json")
    print(f"Use these rules in your training configuration!")