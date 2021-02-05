python produce_systematics.py -i /work/gallim/dataframes/2018_flashgg_UNCORRECTED/FINAL_df_mc_EB_test_corr_clf_5M.h5 \
    -s /work/gallim/dataframes/2018_flashgg_UNCORRECTED/MIX_df_mc_EB_test_corr_clf_5M_spl1.h5 \
    -t /work/gallim/dataframes/2018_flashgg_UNCORRECTED/MIX_df_mc_EB_test_corr_clf_5M_spl2.h5 \
    -d /work/gallim/dataframes/2018_flashgg_UNCORRECTED/FINAL_df_data_EB_test_IdMVA_5M.h5 \
    --factor 2. \
    --shiftF para \
    --outfile /work/gallim/dataframes/2018_flashgg_UNCORRECTED/SystematicsIDMVA_LegRunII_v1_UL2018.root
