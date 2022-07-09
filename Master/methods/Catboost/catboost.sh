python result_with_user.py \
--model catboost \
--feature Fasttext_ave3, Bert_ave3, wordchar, tfidf_ave3, glove_ave3, lsa_ave3, uid, userdata, pathalias, category, other, image_resnext_subcategory_ave5 \
--submission_path ../ \

wait

python result_without_user.py \
--model catboost \
--feature Fasttext_ave3, Bert_ave3, wordchar, tfidf_ave3, glove_ave3, lsa_ave3, uid, userdata, pathalias, category, other, image_resnext_subcategory_ave5 \
--submission_path ../ \

wait 
echo "Over"