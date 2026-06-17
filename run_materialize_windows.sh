# modify for batch processing all the root files
# --root-file-list sbn_anomaly/data/filelist.txt \

# modify for single file testing
# --root-files root://fndcadoor.fnal.gov//pnfs/fs/usr/sbnd/persistent/users/micarrig/DQM//CI_build_lar_ci_19305/reco/reco/DQMValidationTrees_00.root root://fndcadoor.fnal.gov//pnfs/fs/usr/sbnd/persistent/users/micarrig/DQM//CI_build_lar_ci_19305/reco/reco/DQMValidationTrees_01.root\

# running training data materialization
# --root-file-list data/filelist_train.txt \

# running testing data materialization
# --root-file-list data/filelist_test.txt \

python -m sbn_anomaly.data.materialize_windows \
  --config configs/materialize_windows.yaml \
  --root-file-list data/filelist.txt \
  --output testing.npz