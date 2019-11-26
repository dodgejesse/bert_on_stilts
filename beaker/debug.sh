#EXPERIMENT_IDS="ex_cbqaavyg7dl4 ex_kcg7b4dlvwr5 ex_85jct8xq12yt ex_1ywvife93y6n ex_8cne39ee7tb5 ex_j84ja6afqk9a ex_zqnr5y6015ui ex_yd3sx8jliz83 ex_cxq1mzdpa12l ex_vum1f6sixz7v ex_aqziwqwofpwr ex_h9rc1xub10z5 ex_gu36wc4rlnax ex_0820nx4oqtzp ex_kx1yxs1r7fbn ex_bombdfij58mp ex_rnne1c8pryu4 ex_cjewmlrofq5g ex_utk57od7vdln ex_d0o0c3xgwh9a ex_ia5wo4cah9vy ex_4k1v3jgqmjnx ex_stsrafsxj82v ex_r7cu2luyplpl ex_df9r7oep45kf ex_hnm9cggp1txj ex_iyh6l7w12dku ex_a3sp65weqg1v ex_q58w7k829yx5 ex_4gtf0g88qdrh ex_5eub6x4baazb ex_pauus9cjybf0 ex_i9zsrmiir8gn ex_0si72beurwjx ex_spzsp3lqohb1 ex_z8i0zzo3pywt ex_ctep1pqs8tvm ex_488revp4q53u ex_ee07rv1kcys1 ex_b53rj65rcx5r ex_ejhl29ww1amx ex_aolsi600uqu3 ex_3hr8opr4eaig ex_cw9ys2qz4p7u ex_3du14kwb56gu ex_nbd5mu87a5ml ex_lr1zoitkonfr ex_n1l05av80vxd ex_zrauy0utzh4g ex_qxvg6cws2ds4 ex_zy9k7hi1a1i2 ex_q0zlj70y3ayx ex_j8acsa4y7eor ex_4ngcbqpaufp7 ex_4u24bfwz61yy ex_wde4j9k144af ex_pfdknbgfd3es ex_bw67ss3kym84 ex_wtadhglgs7be ex_q8r2n4dki7qg ex_fvys83urj648 ex_u55jc5xpbho2 ex_no0kzdpxsmgv ex_8kc1w7jztrih ex_57zdo0t3ctox ex_tykxu15b4ql6 ex_534k0jr96eus ex_vmx8mx5mbowx ex_vcml6dqhjzio ex_olumqc5iogoi ex_te7ygfa9xsx0 ex_hwqsn787oyqk ex_zw8s8z3ijyyc ex_du8uuqflxhu2 ex_jlns6ovn7amx ex_bgwfsno2y6gv ex_4ewxjajz4v1v ex_4w21qtqrplw3 ex_3rmxte84qerr ex_ly1iqx2vufxg ex_mv5aym5jpkjo ex_mzx3qu9b5uf1 ex_eqcrda9slby3 ex_o9pyqf5kc8ze ex_jgq6or5uu6uc ex_ee6naghnywgi ex_iat1761xotwb ex_jzsro3zujdm6 ex_7e8ggashojv9 ex_4zs71muc7e00 ex_cb33z5ycep8o ex_2vbujqtnjv6l ex_8v52xumpixcq ex_yf7efg82je1s ex_7359xf41kc0l ex_1gdtsdo9a5bu ex_adlhlbbaeh7f ex_7ytsjpqjp6qx ex_b8wkwmd4u9kl ex_kx7cjy412qwm"


EXPERIMENT_IDS="ex_3ytt7xoipsi6"
bash beaker-get-results-dataset.sh ${EXPERIMENT_IDS} mrpc

exit
for CUR_ID in ${EXPERIMENT_IDS}; do
    bash beaker-get-results-dataset.sh ${CUR_ID} ${TASK}
    sleep 3
done


exit













for INIT_SEED in {1..10}; do
    for DATA_SEED in {11..15}; do
	echo "${INIT_SEED}, ${DATA_SEED}"
    done
done
for INIT_SEED in {11..15}; do
    for DATA_SEED in {1..10}; do
	echo "${INIT_SEED}, ${DATA_SEED}"
    done
done

exit





for PAIR in "1 6" "1 8" "2 1" "2 10" "3 1" "3 3" "4 2" "4 6" "5 9" "6 3" "7 3" "7 4" "7 10" "8 2" "8 5" "9 3" "9 5" "9 8" "10 1" "10 4" "10 5" "10 10"; do
    VAR1=$(echo $PAIR | cut -f1 -d " ")
    VAR2=$(echo $PAIR | cut -f2 -d " ")

    
    echo "$PAIR and $VAR1 and $VAR2"
    
done
