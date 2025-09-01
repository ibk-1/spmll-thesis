#!/usr/bin/env python3
"""
Simple Example: Using Logical Constraints with SPMLL

This script shows a practical example of how to integrate logical constraints
into your existing SPMLL training pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from logical_constraints_integration import (
    LogicalConstraints, 
    ConstraintAwareLoss, 
    ConstraintMetrics,
    ConstraintAwareTrainer
)

def create_simple_model(num_classes, input_dim=128):
    """Create a simple neural network for demonstration"""
    model = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes),
        nn.Sigmoid()
    )
    return model

def create_synthetic_batch(batch_size=32, num_classes=5):
    """Create a synthetic batch for demonstration"""
    # Random features
    features = torch.randn(batch_size, 128)
    
    # Random labels (0 or 1)
    labels = torch.randint(0, 2, (batch_size, num_classes), dtype=torch.float32)
    
    # Ensure at least one positive label per sample
    for i in range(batch_size):
        if labels[i].sum() == 0:
            labels[i, torch.randint(0, num_classes, (1,))] = 1
    
    return {
        'image': features,  # Using 'image' key for compatibility
        'label_vec_obs': labels,
        'preds': torch.sigmoid(torch.randn(batch_size, num_classes))  # Random predictions
    }

def demonstrate_constraints():
    """Demonstrate how to use logical constraints"""
    print("=" * 60)
    print("Logical Constraints Demonstration")
    print("=" * 60)
    
    # 1. Define constraints for a 5-class problem
    num_classes = 5
    constraints = LogicalConstraints(num_classes)
    
    # Add some realistic constraints
    constraints.add_mutual_exclusion(0, 1)  # Class 0 and 1 cannot coexist
    constraints.add_implication(2, 3)        # If class 2 is present, class 3 must be present
    constraints.add_hierarchical(4, 3)       # Class 4 is parent of class 3
    constraints.add_cardinality_max(3)       # Maximum 3 classes can be present
    constraints.add_cardinality_min(1)       # At least 1 class must be present
    
    print("✓ Constraints defined:")
    print(constraints)
    print(f"Total constraints: {len(constraints.constraints)}")
    
    # 2. Create a simple model
    model = create_simple_model(num_classes)
    print(f"\n✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 3. Create constraint-aware loss
    def dummy_loss_fn(batch, P, Z):
        """Simple loss function for demonstration"""
        preds = batch['preds']
        labels = batch['label_vec_obs']
        loss = nn.BCELoss()(preds, labels)
        return loss, None
    
    constraint_loss_fn = ConstraintAwareLoss(
        dummy_loss_fn, 
        constraints, 
        constraint_weight=0.1
    )
    print("✓ Constraint-aware loss function created")
    
    # 4. Create constraint metrics
    constraint_metrics = ConstraintMetrics(constraints)
    print("✓ Constraint metrics object created")
    
    # 5. Test with synthetic data
    print("\n" + "-" * 40)
    print("Testing with synthetic data...")
    
    batch = create_synthetic_batch(batch_size=16, num_classes=num_classes)
    
    # Compute constraint-aware loss
    loss, reg_loss = constraint_loss_fn(batch, {}, {})
    print(f"Constraint-aware loss: {loss.item():.4f}")
    
    # Compute constraint metrics
    metrics = constraint_metrics.compute_metrics(batch['preds'])
    print(f"Constraint satisfaction rate: {metrics['overall_satisfaction_rate']:.4f}")
    print(f"Average violations per sample: {metrics['avg_violations_per_sample']:.4f}")
    
    # 6. Show how to integrate with training
    print("\n" + "-" * 40)
    print("Training integration example...")
    
    # Create trainer
    trainer = ConstraintAwareTrainer(model, constraints, constraint_weight=0.1)
    print("✓ Constraint-aware trainer created")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Simulate training for a few steps
    print("\nSimulating training...")
    for step in range(5):
        batch = create_synthetic_batch(batch_size=8, num_classes=num_classes)
        
        # Forward pass
        optimizer.zero_grad()
        preds = model(batch['image'])
        batch['preds'] = preds
        
        # Compute loss
        loss, _ = constraint_loss_fn(batch, {}, {})
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"Step {step + 1}: Loss = {loss.item():.4f}")
    
    print("\n✓ Training simulation completed!")
    
    return constraints, model, constraint_loss_fn, constraint_metrics

def show_integration_with_existing_code():
    """Show how to integrate with existing SPMLL code"""
    print("\n" + "=" * 60)
    print("Integration with Existing SPMLL Code")
    print("=" * 60)
    
    print("To integrate constraints with your existing SPMLL code:")
    print()
    print("1. Import the constraint classes:")
    print("   from logical_constraints_integration import LogicalConstraints, ConstraintAwareLoss")
    print()
    print("2. Define constraints for your dataset:")
    print("   constraints = LogicalConstraints(num_classes=your_num_classes)")
    print("   constraints.add_mutual_exclusion(0, 1)")
    print("   constraints.add_implication(2, 3)")
    print()
    print("3. Replace loss computation in train.py:")
    print("   # OLD: batch = compute_batch_loss(batch, P, Z)")
    print("   # NEW: batch = constraint_loss_fn(batch, P, Z)")
    print()
    print("4. Add constraint metrics to evaluation:")
    print("   constraint_metrics = ConstraintMetrics(constraints)")
    print("   metrics = constraint_metrics.compute_metrics(predictions)")
    print()
    print("5. Adjust constraint weights as needed:")
    print("   constraint_weight=0.1  # Start small, increase if needed")

def main():
    """Main function"""
    try:
        # Demonstrate constraints
        constraints, model, constraint_loss_fn, constraint_metrics = demonstrate_constraints()
        
        # Show integration guide
        show_integration_with_existing_code()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nYou can now:")
        print("1. Run this script to see constraints in action")
        print("2. Modify constraints for your specific dataset")
        print("3. Integrate with your existing SPMLL training code")
        print("4. Experiment with different constraint weights")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("This might be due to missing dependencies or import issues.")
        print("Make sure you have PyTorch installed and the CCN project in the parent directory.")

if __name__ == "__main__":
    main()
