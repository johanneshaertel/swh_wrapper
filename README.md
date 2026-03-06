# RESULTS

(5 folds on the herbod data with 30% test set; first all_bugs is sanity check.)

all_bug (4450 commits, 209540 lines, 5 folds)
  precision : 0.312  0.328  0.313  0.318  0.301  mean=0.314 std=0.009
  recall    : 1.000  1.000  1.000  1.000  1.000  mean=1.000 std=0.000
  f1        : 0.475  0.494  0.477  0.483  0.463  mean=0.478 std=0.010
  mcc       : 0.000  0.000  0.000  0.000  0.000  mean=0.000 std=0.000

heuristic (4450 commits, 209540 lines, 5 folds)
  precision : 0.811  0.809  0.818  0.807  0.801  mean=0.809 std=0.006
  recall    : 0.935  0.948  0.954  0.933  0.945  mean=0.943 std=0.008
  f1        : 0.869  0.873  0.880  0.866  0.867  mean=0.871 std=0.005
  mcc       : 0.807  0.810  0.825  0.801  0.808  mean=0.810 std=0.008

ml_logreg (4450 commits, 209540 lines, 5 folds)
  precision : 0.844  0.841  0.839  0.835  0.815  mean=0.835 std=0.011
  recall    : 0.955  0.969  0.970  0.960  0.965  mean=0.964 std=0.006
  f1        : 0.896  0.901  0.900  0.893  0.883  mean=0.895 std=0.006
  mcc       : 0.848  0.852  0.854  0.843  0.833  mean=0.846 std=0.007

ml_rf (4450 commits, 209540 lines, 5 folds)
  precision : 0.836  0.840  0.837  0.830  0.817  mean=0.832 std=0.008
  recall    : 0.935  0.944  0.950  0.928  0.941  mean=0.940 std=0.008
  f1        : 0.883  0.889  0.890  0.877  0.874  mean=0.883 std=0.006
  mcc       : 0.829  0.834  0.839  0.817  0.819  mean=0.827 std=0.008