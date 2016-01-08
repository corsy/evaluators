import numpy as np
import roc_evaluator as roc

# Evaluate the binary results
y_true = np.array([1, 0, 1, 1, 0, 1, 1, 1])
y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 1])

roc.draw_roc_curve_binary(y_true, y_pred)

# # Evaluate Multiple class with macro avg. of precision and recall
# # 3 Classes:
# y_true = np.array([0, 1, 0, 0, 2])
# # prediction probability for each class
# y_pred = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.6, 0.2, 0.2], [0.4, 0.5, 0.1], [0.3, 0.6, 0.1]])
# roc.draw_roc_curve_multiple(y_true, y_pred)