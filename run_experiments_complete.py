"""
Complete Automated Experiment Pipeline
=======================================

Runs comprehensive experiments:
1. Rule mining from label data
2. Baseline (no constraints)
3. Łukasiewicz fuzzy logic + ablations
4. Product t-norm fuzzy logic + ablations

Everything logged to W&B for thesis plots.
"""

import os
import sys
import copy
import json
import wandb
import datetime
import numpy as np
from pathlib import Path
import torch

# Import training modules (your existing code)
from train import execute_training_run

# Import new modules
from rule_mining import mine_rules_for_dataset, PASCAL_CLASSES, COCO_CLASSES
from fuzzy_penalties import compute_fuzzy_constraints_loss, compute_violation_metrics


# ============================================================================
# BASE CONFIGURATION
# ============================================================================

def get_base_config(dataset='pascal', loss='role'):
    """Get base configuration for experiments."""
    
    feat_dims = {"resnet50": 2048}
    expected_pos = {"pascal": 1.5, "coco": 2.9, "nuswide": 1.9, "cub": 31.4}
    
    linear_init_params = {
        "role": {
            "pascal": {"linear_init_lr": 1e-3, "linear_init_bsize": 16},
            "coco": {"linear_init_lr": 1e-3, "linear_init_bsize": 16},
        },
    }
    
    P = {
        'dataset': dataset,
        'loss': loss,
        'train_mode': 'linear_init',
        'val_set_variant': 'clean',
        'load_path': './data',
        'save_path': './results',
        'lr_mult': 10.0,
        'stop_metric': 'map',
        'ls_coef': 0.1,
        'seed': 1200,
        'use_pretrained': True,
        'num_workers': 4,
        'split_seed': 1200,
        'val_frac': 0.2,
        'ss_seed': 999,
        'ss_frac_train': 1.0,
        'ss_frac_val': 1.0,
        'feature_extractor_arch': 'resnet50',
    }
    
    # Loss-dependent
    P['train_set_variant'] = 'observed' if loss == 'role' else 'clean'
    
    # Training mode
    if P['train_mode'] == 'linear_init':
        P['num_epochs'] = 25
        P['freeze_feature_extractor'] = True
        P['use_feats'] = True
        P['arch'] = 'linear'
        P['linear_init_lr'] = linear_init_params[loss][dataset]['linear_init_lr']
        P['linear_init_bsize'] = linear_init_params[loss][dataset]['linear_init_bsize']
        P['bsize'] = P['linear_init_bsize']
        P['lr'] = P['linear_init_lr']
    
    # Dataset-dependent
    P['feat_dim'] = feat_dims[P['feature_extractor_arch']]
    P['expected_num_pos'] = expected_pos[dataset]
    P['train_feats_file'] = f"./data/{dataset}/train_features_imagenet_{P['feature_extractor_arch']}.npy"
    P['val_feats_file'] = f"./data/{dataset}/val_features_imagenet_{P['feature_extractor_arch']}.npy"
    
    return P


# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

def generate_complete_experiments(dataset, impl_rules, excl_rules):
    """
    Generate complete experiment suite:
    - 1 baseline
    - 6 Łukasiewicz (2 standard + 4 ablations)
    - 6 Product (2 standard + 4 ablations)
    
    Total: 13 experiments per dataset
    """
    experiments = []
    
    # ========================================================================
    # 1. BASELINE (No Constraints)
    # ========================================================================
    
    baseline = get_base_config(dataset, 'role')
    baseline.update({
        'experiment_name': f'{dataset}_baseline',
        'use_constraints': False,
        'fuzzy_type': None,
        'lambda_constraints': 0.0,
    })
    experiments.append((f'{dataset.upper()} - Baseline', baseline))
    
    # ========================================================================
    # 2. ŁUKASIEWICZ T-NORM EXPERIMENTS
    # ========================================================================
    
    # 2a. Łukasiewicz - Moderate
    luk_moderate = get_base_config(dataset, 'role')
    luk_moderate.update({
        'experiment_name': f'{dataset}_lukasiewicz_moderate',
        'use_constraints': True,
        'fuzzy_type': 'lukasiewicz',
        'lambda_constraints': 0.5,
        'lambda_impl': 1.0,
        'lambda_excl': 1.0,
        'constraints_warmup_epochs': 3,
    })
    experiments.append((f'{dataset.upper()} - Łukasiewicz Moderate', luk_moderate))
    
    # 2b. Łukasiewicz - Strong
    luk_strong = get_base_config(dataset, 'role')
    luk_strong.update({
        'experiment_name': f'{dataset}_lukasiewicz_strong',
        'use_constraints': True,
        'fuzzy_type': 'lukasiewicz',
        'lambda_constraints': 1.0,
        'lambda_impl': 1.5,
        'lambda_excl': 1.5,
        'constraints_warmup_epochs': 5,
    })
    experiments.append((f'{dataset.upper()} - Łukasiewicz Strong', luk_strong))
    
    # 2c-e. Łukasiewicz - Ablations (lambda_constraints)
    for lam_c in [0.1, 0.3, 0.8]:
        luk_abl = get_base_config(dataset, 'role')
        luk_abl.update({
            'experiment_name': f'{dataset}_lukasiewicz_lambda{lam_c}',
            'use_constraints': True,
            'fuzzy_type': 'lukasiewicz',
            'lambda_constraints': lam_c,
            'lambda_impl': 1.0,
            'lambda_excl': 1.0,
            'constraints_warmup_epochs': 3,
        })
        experiments.append((f'{dataset.upper()} - Łukasiewicz λ={lam_c}', luk_abl))
    
    # 2f. Łukasiewicz - Implication Only
    luk_impl = get_base_config(dataset, 'role')
    luk_impl.update({
        'experiment_name': f'{dataset}_lukasiewicz_impl_only',
        'use_constraints': True,
        'fuzzy_type': 'lukasiewicz',
        'lambda_constraints': 0.5,
        'lambda_impl': 1.0,
        'lambda_excl': 0.0,  # Disable
        'constraints_warmup_epochs': 3,
    })
    experiments.append((f'{dataset.upper()} - Łukasiewicz Impl Only', luk_impl))
    
    # ========================================================================
    # 3. PRODUCT T-NORM EXPERIMENTS
    # ========================================================================
    
    # 3a. Product - Moderate
    prod_moderate = get_base_config(dataset, 'role')
    prod_moderate.update({
        'experiment_name': f'{dataset}_product_moderate',
        'use_constraints': True,
        'fuzzy_type': 'product',
        'lambda_constraints': 0.5,
        'lambda_impl': 1.0,
        'lambda_excl': 1.0,
        'constraints_warmup_epochs': 3,
    })
    experiments.append((f'{dataset.upper()} - Product Moderate', prod_moderate))
    
    # 3b. Product - Strong
    prod_strong = get_base_config(dataset, 'role')
    prod_strong.update({
        'experiment_name': f'{dataset}_product_strong',
        'use_constraints': True,
        'fuzzy_type': 'product',
        'lambda_constraints': 1.0,
        'lambda_impl': 1.5,
        'lambda_excl': 1.5,
        'constraints_warmup_epochs': 5,
    })
    experiments.append((f'{dataset.upper()} - Product Strong', prod_strong))
    
    # 3c-e. Product - Ablations (lambda_constraints)
    for lam_c in [0.1, 0.3, 0.8]:
        prod_abl = get_base_config(dataset, 'role')
        prod_abl.update({
            'experiment_name': f'{dataset}_product_lambda{lam_c}',
            'use_constraints': True,
            'fuzzy_type': 'product',
            'lambda_constraints': lam_c,
            'lambda_impl': 1.0,
            'lambda_excl': 1.0,
            'constraints_warmup_epochs': 3,
        })
        experiments.append((f'{dataset.upper()} - Product λ={lam_c}', prod_abl))
    
    # 3f. Product - Implication Only
    prod_impl = get_base_config(dataset, 'role')
    prod_impl.update({
        'experiment_name': f'{dataset}_product_impl_only',
        'use_constraints': True,
        'fuzzy_type': 'product',
        'lambda_constraints': 0.5,
        'lambda_impl': 1.0,
        'lambda_excl': 0.0,
        'constraints_warmup_epochs': 3,
    })
    experiments.append((f'{dataset.upper()} - Product Impl Only', prod_impl))
    
    # Add rules to all experiments
    for _, config in experiments:
        config['implication_rules'] = impl_rules
        config['exclusion_rules'] = excl_rules
    
    return experiments


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_single_experiment(exp_name, config, wandb_project="spmll_thesis", 
                         wandb_entity="ibrahimkaliljh-student"):
    """Run a single experiment with comprehensive W&B logging."""
    
    print("\n" + "="*80)
    print(f"🚀 STARTING: {exp_name}")
    print("="*80)
    
    # Print config summary
    print(f"\n📋 Configuration:")
    print(f"  Dataset: {config['dataset']}")
    print(f"  Loss: {config['loss']}")
    print(f"  Constraints: {config.get('use_constraints', False)}")
    if config.get('use_constraints', False):
        print(f"  Fuzzy Type: {config.get('fuzzy_type', 'N/A')}")
        print(f"  λ_constraints: {config.get('lambda_constraints', 0.0)}")
        print(f"  λ_impl: {config.get('lambda_impl', 1.0)}")
        print(f"  λ_excl: {config.get('lambda_excl', 1.0)}")
        print(f"  Warmup: {config.get('constraints_warmup_epochs', 0)} epochs")
        print(f"  Impl rules: {len(config.get('implication_rules', []))}")
        print(f"  Excl rules: {len(config.get('exclusion_rules', []))}")
    
    # Create save directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(config['save_path'], f"{config['experiment_name']}_{timestamp}")
    config['save_path'] = save_path
    os.makedirs(save_path, exist_ok=True)
    
    # Save config
    config_save = copy.deepcopy(config)
    # Convert lists to JSON-serializable format
    if 'implication_rules' in config_save:
        config_save['implication_rules'] = [[int(a), int(b)] for a, b in config_save['implication_rules']]
    if 'exclusion_rules' in config_save:
        config_save['exclusion_rules'] = [[int(a), int(b)] for a, b in config_save['exclusion_rules']]
    
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(config_save, f, indent=2)
    
    # Initialize W&B
    wandb_run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=exp_name,
        config=config_save,
        reinit=True,
        tags=[
            config['dataset'],
            'baseline' if not config.get('use_constraints') else config.get('fuzzy_type', 'unknown'),
            f"lambda_{config.get('lambda_constraints', 0.0)}"
        ]
    )
    
    try:
        print(f"\n🎓 Training started...")
        
        # Run training
        feature_extractor, linear_classifier, estimated_labels, logs = execute_training_run(
            config,
            feature_extractor=None,
            linear_classifier=None,
            estimated_labels=None
        )
        
        # Extract final results
        results = {
            'experiment_name': exp_name,
            'config': config_save,
            'save_path': save_path,
            'status': 'completed'
        }
        
        # Extract best metrics
        try:
            if 'val' in logs and 'map' in logs['val']:
                val_maps = logs['val']['map']
                best_epoch = int(np.argmax(val_maps))
                best_val_map = float(val_maps[best_epoch])
                
                results['best_epoch'] = best_epoch
                results['best_val_map'] = best_val_map
                
                if 'test' in logs and 'map' in logs['test']:
                    test_map = float(logs['test']['map'][best_epoch])
                    results['test_map_at_best'] = test_map
                
                # Log final summary to W&B
                wandb.run.summary['best_val_map'] = best_val_map
                wandb.run.summary['best_epoch'] = best_epoch
                if 'test_map_at_best' in results:
                    wandb.run.summary['test_map_at_best'] = results['test_map_at_best']
                
                print(f"\n✅ COMPLETED: {exp_name}")
                print(f"📊 Best Val mAP: {best_val_map:.4f} at epoch {best_epoch}")
                if 'test_map_at_best' in results:
                    print(f"📊 Test mAP: {results['test_map_at_best']:.4f}")
        
        except Exception as e:
            print(f"⚠️  Could not extract metrics: {e}")
            results['status'] = 'completed_with_warnings'
        
        return results
        
    except Exception as e:
        print(f"\n❌ FAILED: {exp_name}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'experiment_name': exp_name,
            'status': 'failed',
            'error': str(e)
        }
    
    finally:
        wandb.finish()


def run_all_experiments(experiments, wandb_project="spmll_thesis"):
    """Run all experiments in sequence."""
    
    print("\n" + "#" * 40)
    print(f"RUNNING {len(experiments)} EXPERIMENTS")
    print("#" * 40)
    
    all_results = []
    success_count = 0
    fail_count = 0
    
    for i, (exp_name, config) in enumerate(experiments, 1):
        print(f"\n\n{'#'*80}")
        print(f"EXPERIMENT {i}/{len(experiments)}")
        print(f"{'#'*80}")
        
        result = run_single_experiment(exp_name, config, wandb_project)
        all_results.append(result)
        
        if result['status'] in ['completed', 'completed_with_warnings']:
            success_count += 1
        else:
            fail_count += 1
        
        # Save intermediate results
        results_file = os.path.join('./results', 'all_experiments_results.json')
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
    
    # Final summary
    print("\n\n" + "#" * 40)
    print("ALL EXPERIMENTS COMPLETED!")
    print("#" * 40)
    
    print(f"\nSUMMARY:")
    print(f"  Total: {len(experiments)}")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")

    print("\nRESULTS:")
    print("-" * 80)
    for result in all_results:
        exp_name = result['experiment_name']
        status = result['status']
        
        if status == 'completed':
            val_map = result.get('best_val_map', 'N/A')
            test_map = result.get('test_map_at_best', 'N/A')
            print(f"{exp_name}")
            print(f"   Val mAP: {val_map:.4f} | Test mAP: {test_map:.4f}")
        elif status == 'completed_with_warnings':
            print(f"{exp_name} - completed with warnings")
        else:
            print(f"{exp_name} - FAILED")
    
    print("\n" + "#"*80)
    print(f"Results saved to: ./results/all_experiments_results.json")
    
    return all_results


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main_pipeline(datasets=['pascal', 'coco'], 
                 wandb_project="spmll_thesis",
                 mine_rules=True):
    """
    Complete automated pipeline:
    1. Mine rules (if requested)
    2. Generate experiments
    3. Run all experiments
    """
    
    all_experiments = []
    
    for dataset in datasets:
        print(f"\n{'#'*80}")
        print(f"# PREPARING {dataset.upper()} EXPERIMENTS")
        print(f"{'#'*80}\n")
        
        # Select class names
        class_names = PASCAL_CLASSES if dataset == 'pascal' else COCO_CLASSES
        
        # Mine or load rules
        if mine_rules:
            labels_file = r"data\pascal\formatted_train_labels.npy" if dataset == 'pascal' else r"data\coco\formatted_train_labels.npy"
            
            if not os.path.exists(labels_file):
                print(f"❌ ERROR: {labels_file} not found!")
                print(f"Please provide {dataset}_labels.npy in the current directory.")
                continue
            
            print(f"🔍 Mining rules from {labels_file}...")
            impl_pairs, excl_pairs, miner = mine_rules_for_dataset(
                dataset_name=dataset,
                labels_file=labels_file,
                class_names=class_names,
                impl_confidence=0.7,
                impl_lift=1.5,
                excl_max_cooccur=0.05,
                top_k_impl=50,
                top_k_excl=30
            )
        else:
            # Load from JSON
            rules_file = f"{dataset}_mined_rules.json"
            if not os.path.exists(rules_file):
                print(f"ERROR: {rules_file} not found!")
                continue
            
            with open(rules_file, 'r') as f:
                rules_data = json.load(f)
            
            impl_pairs = [(r['from_class'], r['to_class']) 
                         for r in rules_data['implication_rules']]
            excl_pairs = [(r['class_a'], r['class_b']) 
                         for r in rules_data['exclusion_rules']]
            
            print(f"✓ Loaded {len(impl_pairs)} implication + {len(excl_pairs)} exclusion rules")
        
        # Generate experiments
        print(f"\nGenerating experiment configurations...")
        dataset_experiments = generate_complete_experiments(dataset, impl_pairs, excl_pairs)
        print(f"✓ Generated {len(dataset_experiments)} experiments for {dataset}")
        
        all_experiments.extend(dataset_experiments)
    
    # Run all experiments
    if all_experiments:
        print(f"\nStarting {len(all_experiments)} experiments...")
        results = run_all_experiments(all_experiments, wandb_project)
        return results
    else:
        print("No experiments to run!")
        return []


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run complete experiment pipeline')
    parser.add_argument('--datasets', nargs='+', default=['pascal'],
                       choices=['pascal', 'coco'],
                       help='Datasets to run experiments on')
    parser.add_argument('--wandb-project', type=str, default='spmll_thesis',
                       help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default='ibrahimkaliljh-student',
                       help='W&B entity name')
    parser.add_argument('--mine-rules', action='store_true', default=True,
                       help='Mine rules from data (default: True)')
    parser.add_argument('--use-existing-rules', action='store_true',
                       help='Use existing mined rules (skip mining)')
    
    args = parser.parse_args()
    
    # Override global entity
    import __main__
    __main__.WANDB_ENTITY = args.wandb_entity
    
    print("\n" + "#" * 40)
    print("MASTER THESIS EXPERIMENT PIPELINE")
    print("Fuzzy Logic Constraints for Multi-Label Learning")
    print("#" * 40)
    
    print(f"\nDatasets: {', '.join(args.datasets)}")
    print(f"W&B Project: {args.wandb_project}")
    print(f"Mine Rules: {not args.use_existing_rules}")
    
    # Run pipeline
    results = main_pipeline(
        datasets=args.datasets,
        wandb_project=args.wandb_project,
        mine_rules=not args.use_existing_rules
    )
    
    print("\nPIPELINE COMPLETE!")
    print("\nNext steps:")
    print("1. Check W&B dashboard for training curves")
    print("2. Run analysis: python analysis_utils.py --results-dir ./results")
    print("3. Use plots and tables for your thesis!")