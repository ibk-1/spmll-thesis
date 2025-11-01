# Results: With vs Without Logical Constraints

## Overall Metrics

| Metric                 | Without Constraints | With Constraints | Change  |
|-------------------------|---------------------|-----------------|---------|
| mAP (macro mean AP)    | 0.8712              | 0.8719          | +0.0007 |
| AP (micro)             | 0.9189              | 0.9186          | −0.0003 |
| AUROC (macro)          | 0.9738              | 0.9744          | +0.0006 |
| AUROC (micro)          | 0.9789              | 0.9755          | −0.0034 |
| Precision (micro @0.5) | 0.9482              | 0.9454          | −0.0028 |
| Recall (micro @0.5)    | 0.7211              | 0.7325          | +0.0114 |
| F1 (micro @0.5)        | 0.8192              | 0.8255          | +0.0063 |
| Precision (macro @0.5) | 0.9351              | 0.9319          | −0.0032 |
| Recall (macro @0.5)    | 0.6418              | 0.6427          | +0.0009 |
| F1 (macro @0.5)        | 0.7470              | 0.7477          | +0.0007 |
| Subset Accuracy        | 0.5406              | 0.5597          | +0.0191 |
| Hamming Loss           | 0.0328              | 0.0320          | −0.0009 |

---

## Per-class F1 Score Comparison

| Class        | F1 (Without) | F1 (With) | ΔF1   |
|--------------|--------------|-----------|-------|
| dog          | 0.762        | 0.853     | +0.091 |
| horse        | 0.724        | 0.810     | +0.086 |
| animal       | 0.905        | 0.955     | +0.050 |
| aeroplane    | 0.901        | 0.938     | +0.037 |
| diningtable  | 0.402        | 0.440     | +0.038 |
| cow          | 0.722        | 0.754     | +0.032 |
| car          | 0.708        | 0.738     | +0.031 |
| sofa         | 0.536        | 0.553     | +0.017 |
| pottedplant  | 0.478        | 0.497     | +0.019 |
| cat          | 0.900        | 0.905     | +0.005 |
| bus          | 0.854        | 0.856     | +0.002 |
| vehicle      | 0.924        | 0.916     | −0.008 |
| person       | 0.890        | 0.882     | −0.008 |
| sheep        | 0.822        | 0.807     | −0.015 |
| tvmonitor    | 0.674        | 0.659     | −0.015 |
| bottle       | 0.521        | 0.507     | −0.014 |
| bird         | 0.900        | 0.867     | −0.033 |
| indoor       | 0.736        | 0.705     | −0.031 |
| boat         | 0.803        | 0.764     | −0.039 |
| train        | 0.886        | 0.839     | −0.047 |
| motorbike    | 0.811        | 0.765     | −0.046 |
| chair        | 0.525        | 0.458     | −0.067 |
| bicycle      | 0.798        | 0.727     | −0.071 |

---

## Summary
- **Animal hierarchy**: clear improvements for both parent and children (dog, horse, animal).  
- **Vehicle hierarchy**: mixed; some subclasses up (aeroplane, car), others down (train, motorbike, bicycle).  
- **Indoor hierarchy**: mostly negative; parent and children lost performance.  
