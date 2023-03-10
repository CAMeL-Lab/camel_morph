camel_tools = "/Users/chriscay/Library/Mobile Documents/com~apple~CloudDocs/NYUAD/camel_tools"
service_account = "/Users/chriscay/.config/gspread/service_account.json"

make_db_pv_msa:
	python camel_morph/db_maker.py -config_file configs/config.json -config_name pv_msa_red 
make_db_iv_msa:
	python camel_morph/db_maker.py -config_file configs/config.json -config_name iv_msa_red 
make_db_cv_msa:	
	python camel_morph/db_maker.py -config_file configs/config.json -config_name cv_msa_red 
make_db_all_aspects_msa_red:
	python camel_morph/db_maker.py -config_file configs/config.json -config_name all_aspects_msa_red 
make_db_all_aspects_msa:
	python camel_morph/db_maker.py -config_file configs/config.json -config_name all_aspects_msa 
make_db_cr_msa:
	python camel_morph/db_maker.py -config_file configs/config.json -config_name msa_cam_ready_sigmorphon2022

make_db_pv_glf:	
	python camel_morph/db_maker.py -config_file configs/config.json -config_name pv_glf_red 
make_db_iv_glf:	
	python camel_morph/db_maker.py -config_file configs/config.json -config_name iv_glf_red 
make_db_cv_glf:	
	python camel_morph/db_maker.py -config_file configs/config.json -config_name cv_glf_red 

make_db_pv_egy:	
	python camel_morph/db_maker.py -config_file configs/config.json -config_name pv_egy_red 
make_db_iv_egy:	
	python camel_morph/db_maker.py -config_file configs/config.json -config_name iv_egy_red 
make_db_cv_egy:	
	python camel_morph/db_maker.py -config_file configs/config.json -config_name cv_egy_red 
make_db_all_aspects_egy:	
	python camel_morph/db_maker.py -config_file configs/config.json -config_name all_aspects_egy 
make_db_cr_egy:	
	python camel_morph/db_maker.py -config_file configs/config.json -config_name egy_cam_ready_sigmorphon2022

make_db_adj_msa:	
	python camel_morph/db_maker.py -config_file configs/config.json -config_name adj_msa_split_red
make_db_noun_msa:	
	python camel_morph/db_maker.py -config_file configs/config.json -config_name noun_msa_split_red
make_db_noun_prop_msa:	
	python camel_morph/db_maker.py -config_file configs/config.json -config_name noun_prop_msa_split_red
make_db_other_msa:
	python camel_morph/db_maker.py -config_file configs/config.json -config_name other_msa_split_red

make_db_adj_egy:
	python camel_morph/db_maker.py -config_file configs/config.json -config_name adj_egy_split_red
make_db_noun_egy:
	python camel_morph/db_maker.py -config_file configs/config.json -config_name noun_egy_split_red
make_db_noun_prop_egy:
	python camel_morph/db_maker.py -config_file configs/config.json -config_name noun_prop_egy_split_red
make_db_other_egy:
	python camel_morph/db_maker.py -config_file configs/config.json -config_name other_egy_split_red

make_db_nom_glf:
	python camel_morph/db_maker.py -config_file configs/config_capstones.json -config_name nom_glf_split_red

repr_lemmas_pv_msa:
	python camel_morph/debugging/create_repr_lemmas_list.py -config_file configs/config.json -config_name pv_msa -display_format unique -feats "asp:p mod:i" -db eval_files/calima-msa-s31_0.4.2.utf8.db
repr_lemmas_iv_msa:	
	python camel_morph/debugging/create_repr_lemmas_list.py -config_file configs/config.json -config_name iv_msa -display_format unique -feats "asp:i mod:i" -db eval_files/calima-msa-s31_0.4.2.utf8.db
repr_lemmas_cv_msa:	
	python camel_morph/debugging/create_repr_lemmas_list.py -config_file configs/config.json -config_name cv_msa -display_format unique -feats "asp:c mod:i" -db eval_files/calima-msa-s31_0.4.2.utf8.db

repr_lemmas_pv_glf:	
	python camel_morph/debugging/create_repr_lemmas_list.py -config_file configs/config.json -config_name pv_glf_red -display_format unique -feats "asp:p mod:i"
repr_lemmas_iv_glf:	
	python camel_morph/debugging/create_repr_lemmas_list.py -config_file configs/config.json -config_name iv_glf_red -display_format unique -feats "asp:i mod:i"
repr_lemmas_cv_glf:	
	python camel_morph/debugging/create_repr_lemmas_list.py -config_file configs/config.json -config_name cv_glf_red -display_format unique -feats "asp:c mod:i"

repr_lemmas_pv_egy:	
	python camel_morph/debugging/create_repr_lemmas_list.py -config_file configs/config.json -config_name pv_egy -display_format unique -feats "asp:p mod:i"
repr_lemmas_iv_egy:	
	python camel_morph/debugging/create_repr_lemmas_list.py -config_file configs/config.json -config_name iv_egy -display_format unique -feats "asp:i mod:i"
repr_lemmas_cv_egy:	
	python camel_morph/debugging/create_repr_lemmas_list.py -config_file configs/config.json -config_name cv_egy -display_format unique -feats "asp:c mod:i"

repr_lemmas_adj_msa:
	python camel_morph/debugging/create_repr_lemmas_list.py -config_file configs/config.json -config_name adj_msa_split_red -display_format unique -db eval_files/calima-msa-s31_0.4.2.utf8.db
repr_lemmas_noun_msa:
	python camel_morph/debugging/create_repr_lemmas_list.py -config_file configs/config.json -config_name noun_msa_split_red adj_msa_split_red -display_format unique -db eval_files/calima-msa-s31_0.4.2.utf8.db
repr_lemmas_noun_prop_msa:
	python camel_morph/debugging/create_repr_lemmas_list.py -config_file configs/config.json -config_name noun_prop_msa_split_red -display_format unique -db eval_files/calima-msa-s31_0.4.2.utf8.db
repr_lemmas_other_msa:
	python camel_morph/debugging/create_repr_lemmas_list.py -config_file configs/config.json -config_name other_msa_split_red -display_format unique -db eval_files/calima-msa-s31_0.4.2.utf8.db

repr_lemmas_adj_egy:
	python camel_morph/debugging/create_repr_lemmas_list.py -config_file configs/config.json -config_name adj_egy_split_red -display_format unique
repr_lemmas_noun_egy:
	python camel_morph/debugging/create_repr_lemmas_list.py -config_file configs/config.json -config_name noun_egy_split_red adj_egy_split_red -display_format unique
repr_lemmas_noun_prop_egy:
	python camel_morph/debugging/create_repr_lemmas_list.py -config_file configs/config.json -config_name noun_prop_egy_split_red -display_format unique
repr_lemmas_other_egy:
	python camel_morph/debugging/create_repr_lemmas_list.py -config_file configs/config.json -config_name other_egy_split_red -display_format unique

repr_lemmas_nom_glf:
	python camel_morph/debugging/create_repr_lemmas_list.py -config_file configs/config_capstones.json -config_name nom_glf_split_red -display_format unique


conj_pv_msa:
	python camel_morph/debugging/generate_conj_table.py -feats "asp:p mod:i" -config_file configs/config.json -config_name pv_msa

conj_iv_i_msa:	
	python camel_morph/debugging/generate_conj_table.py -feats "asp:p mod:i" -config_file configs/config.json -config_name iv_msa
conj_iv_s_msa:	
	python camel_morph/debugging/generate_conj_table.py -feats "asp:i mod:s" -config_file configs/config.json -config_name iv_msa
conj_iv_j_msa:	
	python camel_morph/debugging/generate_conj_table.py -feats "asp:i mod:j" -config_file configs/config.json -config_name iv_msa
conj_iv_e_msa:	
	python camel_morph/debugging/generate_conj_table.py -feats "asp:i mod:e" -config_file configs/config.json -config_name iv_msa
conj_iv_x_msa:	
	python camel_morph/debugging/generate_conj_table.py -feats "asp:i mod:x" -config_file configs/config.json -config_name iv_msa

conj_cv_i_msa:	
	python camel_morph/debugging/generate_conj_table.py -feats "asp:c mod:i" -config_file configs/config.json -config_name cv_msa
conj_cv_e_msa:	
	python camel_morph/debugging/generate_conj_table.py -feats "asp:c mod:e" -config_file configs/config.json -config_name cv_msa
conj_cv_x_msa:	
	python camel_morph/debugging/generate_conj_table.py -feats "asp:c mod:x" -config_file configs/config.json -config_name cv_msa

conj_pv_msa_db_full:
	python camel_morph/debugging/generate_conj_table.py -feats "asp:p mod:i" -config_file configs/config.json -config_name pv_msa -db msa_cam_ready_sigmorphon2022_v1.0.db

conj_iv_i_msa_db_full:	
	python camel_morph/debugging/generate_conj_table.py -feats "asp:i mod:i" -config_file configs/config.json -config_name iv_msa -db msa_cam_ready_sigmorphon2022_v1.0.db
conj_iv_s_msa_db_full:	
	python camel_morph/debugging/generate_conj_table.py -feats "asp:i mod:s" -config_file configs/config.json -config_name iv_msa -db msa_cam_ready_sigmorphon2022_v1.0.db
conj_iv_j_msa_db_full:	
	python camel_morph/debugging/generate_conj_table.py -feats "asp:i mod:j" -config_file configs/config.json -config_name iv_msa -db msa_cam_ready_sigmorphon2022_v1.0.db
conj_iv_e_msa_db_full:	
	python camel_morph/debugging/generate_conj_table.py -feats "asp:i mod:e" -config_file configs/config.json -config_name iv_msa -db msa_cam_ready_sigmorphon2022_v1.0.db
conj_iv_x_msa_db_full:	
	python camel_morph/debugging/generate_conj_table.py -feats "asp:i mod:x" -config_file configs/config.json -config_name iv_msa -db msa_cam_ready_sigmorphon2022_v1.0.db

conj_cv_i_msa_db_full:	
	python camel_morph/debugging/generate_conj_table.py -feats "asp:c mod:i" -config_file configs/config.json -config_name cv_msa -db msa_cam_ready_sigmorphon2022_v1.0.db
conj_cv_e_msa_db_full:	
	python camel_morph/debugging/generate_conj_table.py -feats "asp:c mod:e" -config_file configs/config.json -config_name cv_msa -db msa_cam_ready_sigmorphon2022_v1.0.db
conj_cv_x_msa_db_full:	
	python camel_morph/debugging/generate_conj_table.py -feats "asp:c mod:x" -config_file configs/config.json -config_name cv_msa -db msa_cam_ready_sigmorphon2022_v1.0.db

conj_adj_msa:	
	python camel_morph/debugging/generate_conj_table.py -config_file configs/config.json -config_name adj_msa_split_red
conj_noun_msa:	
	python camel_morph/debugging/generate_conj_table.py -config_file configs/config.json -config_name noun_msa_split_red
conj_noun_prop_msa:	
	python camel_morph/debugging/generate_conj_table.py -config_file configs/config.json -config_name noun_prop_msa_split_red
conj_other_msa:	
	python camel_morph/debugging/generate_conj_table.py -config_file configs/config.json -config_name other_msa_split_red

conj_pv_glf:
	python camel_morph/debugging/generate_conj_table.py -feats "asp:p mod:i" -config_file configs/config.json -config_name pv_glf_red
conj_iv_glf:	
	python camel_morph/debugging/generate_conj_table.py -feats "asp:i mod:i" -config_file configs/config.json -config_name iv_glf_red
conj_cv_glf:	
	python camel_morph/debugging/generate_conj_table.py -feats "asp:c mod:i" -config_file configs/config.json -config_name cv_glf_red

conj_pv_egy:
	python camel_morph/debugging/generate_conj_table.py -feats "asp:p mod:i" -config_file configs/config.json -config_name pv_egy
conj_iv_egy:	
	python camel_morph/debugging/generate_conj_table.py -feats "asp:i mod:i" -config_file configs/config.json -config_name iv_egy
conj_cv_egy:	
	python camel_morph/debugging/generate_conj_table.py -feats "asp:c mod:i" -config_file configs/config.json -config_name iv_egy

conj_adj_egy:	
	python camel_morph/debugging/generate_conj_table.py -config_file configs/config.json -config_name adj_egy_split_red
conj_noun_egy:	
	python camel_morph/debugging/generate_conj_table.py -config_file configs/config.json -config_name noun_egy_split_red
conj_noun_prop_egy:	
	python camel_morph/debugging/generate_conj_table.py -config_file configs/config.json -config_name noun_prop_egy_split_red
conj_other_egy:	
	python camel_morph/debugging/generate_conj_table.py -config_file configs/config.json -config_name other_egy_split_red

conj_nom_glf:
	python camel_morph/debugging/generate_conj_table.py -config_file configs/config_capstones.json -config_name nom_glf_split_red

conj_pv_egy_db_full:
	python camel_morph/debugging/generate_conj_table.py -feats "asp:p mod:i" -config_file configs/config.json -config_name pv_egy -db XYZ_egy_cr_v1.0.db
conj_iv_egy_db_full:	
	python camel_morph/debugging/generate_conj_table.py -feats "asp:i mod:i" -config_file configs/config.json -config_name iv_egy -db XYZ_egy_cr_v1.0.db
conj_cv_egy_db_full:	
	python camel_morph/debugging/generate_conj_table.py -feats "asp:c mod:i" -config_file configs/config.json -config_name cv_egy -db XYZ_egy_cr_v1.0.db

download_specs:
	python camel_morph/debugging/download_sheets.py -specs header-morph-order-sheets MSA-Verb-MORPH
download_msa_pv:	
	python camel_morph/debugging/download_sheets.py -config_file configs/config.json -config_name pv_msa
download_msa_iv:
	python camel_morph/debugging/download_sheets.py -config_file configs/config.json -config_name iv_msa
download_msa_cv:	
	python camel_morph/debugging/download_sheets.py -config_file configs/config.json -config_name cv_msa
download_msa_all_red:	
	python camel_morph/debugging/download_sheets.py -config_file configs/config.json -config_name all_aspects_msa_red
download_msa_all:	
	python camel_morph/debugging/download_sheets.py -config_file configs/config.json -config_name all_aspects_msa
download_msa_cr:
	python camel_morph/debugging/download_sheets.py -config_file configs/config.json -config_name msa_cam_ready_sigmorphon2022

download_glf_pv:	
	python camel_morph/debugging/download_sheets.py -config_file configs/config.json -config_name pv_glf_red
download_glf_iv:	
	python camel_morph/debugging/download_sheets.py -config_file configs/config.json -config_name iv_glf_red
download_glf_cv:	
	python camel_morph/debugging/download_sheets.py -config_file configs/config.json -config_name cv_glf_red

download_egy_pv:	
	python camel_morph/debugging/download_sheets.py -config_file configs/config.json -config_name pv_egy
download_egy_iv:	
	python camel_morph/debugging/download_sheets.py -config_file configs/config.json -config_name iv_egy
download_egy_cv:	
	python camel_morph/debugging/download_sheets.py -config_file configs/config.json -config_name cv_egy
download_egy_all:	
	python camel_morph/debugging/download_sheets.py -config_file configs/config.json -config_name all_aspects_egy
download_cr_egy:	
	python camel_morph/debugging/download_sheets.py -config_file configs/config.json -config_name egy_cam_ready_sigmorphon2022

download_msa_adj:	
	python camel_morph/debugging/download_sheets.py -config_file configs/config.json -config_name adj_msa_split_red
download_msa_noun:	
	python camel_morph/debugging/download_sheets.py -config_file configs/config.json -config_name noun_msa_split_red
download_msa_noun_prop:	
	python camel_morph/debugging/download_sheets.py -config_file configs/config.json -config_name noun_prop_msa_split_red
download_msa_other:	
	python camel_morph/debugging/download_sheets.py -config_file configs/config.json -config_name other_msa_split_red

download_egy_adj:	
	python camel_morph/debugging/download_sheets.py -config_file configs/config.json -config_name adj_egy_split_red
download_egy_noun:	
	python camel_morph/debugging/download_sheets.py -config_file configs/config.json -config_name noun_egy_split_red
download_egy_noun_prop:	
	python camel_morph/debugging/download_sheets.py -config_file configs/config.json -config_name noun_prop_egy_split_red
download_egy_other:	
	python camel_morph/debugging/download_sheets.py -config_file configs/config.json -config_name other_egy_split_red

download_glf_nom:
	python camel_morph/debugging/download_sheets.py -config_file configs/config_capstones.json -config_name nom_glf_split_red

upload_pv_msa:
	python camel_morph/debugging/upload_sheets.py -input_dir tables_dir -config_file configs/config.json -config_name pv_msa -feats "asp:p mod:i"

upload_iv_i_msa:
	python camel_morph/debugging/upload_sheets.py -input_dir tables_dir -config_file configs/config.json -config_name iv_msa -feats "asp:i mod:i"
upload_iv_s_msa:
	python camel_morph/debugging/upload_sheets.py -input_dir tables_dir -config_file configs/config.json -config_name iv_msa -feats "asp:i mod:s"
upload_iv_j_msa:
	python camel_morph/debugging/upload_sheets.py -input_dir tables_dir -config_file configs/config.json -config_name iv_msa -feats "asp:i mod:j"
upload_iv_e_msa:
	python camel_morph/debugging/upload_sheets.py -input_dir tables_dir -config_file configs/config.json -config_name iv_msa -feats "asp:i mod:e"
upload_iv_x_msa:
	python camel_morph/debugging/upload_sheets.py -input_dir tables_dir -config_file configs/config.json -config_name iv_msa -feats "asp:i mod:x"

upload_cv_i_msa:
	python camel_morph/debugging/upload_sheets.py -input_dir tables_dir -config_file configs/config.json -config_name cv_msa -feats "asp:c mod:i"
upload_cv_e_msa:
	python camel_morph/debugging/upload_sheets.py -input_dir tables_dir -config_file configs/config.json -config_name cv_msa -feats "asp:c mod:e"
upload_cv_x_msa:
	python camel_morph/debugging/upload_sheets.py -input_dir tables_dir -config_file configs/config.json -config_name cv_msa -feats "asp:c mod:x"

upload_adj_msa:
	python camel_morph/debugging/upload_sheets.py -input_dir tables_dir -config_file configs/config.json -config_name adj_msa_split_red -mode backup
upload_noun_msa:
	python camel_morph/debugging/upload_sheets.py -input_dir tables_dir -config_file configs/config.json -config_name noun_msa_split_red -mode backup
upload_noun_prop_msa:
	python camel_morph/debugging/upload_sheets.py -input_dir tables_dir -config_file configs/config.json -config_name noun_prop_msa_split_red -mode backup
upload_other_msa:
	python camel_morph/debugging/upload_sheets.py -input_dir tables_dir -config_file configs/config.json -config_name other_msa_split_red -mode backup

upload_pv_glf:
	python camel_morph/debugging/upload_sheets.py -input_dir tables_dir -config_file configs/config.json -config_name pv_glf_red -feats "asp:p mod:i" -mode backup
upload_iv_glf:
	python camel_morph/debugging/upload_sheets.py -input_dir tables_dir -config_file configs/config.json -config_name iv_glf_red -feats "asp:i mod:i" -mode backup
upload_cv_glf:
	python camel_morph/debugging/upload_sheets.py -input_dir tables_dir -config_file configs/config.json -config_name cv_glf_red -feats "asp:c mod:i" -mode backup

upload_pv_egy:
	python camel_morph/debugging/upload_sheets.py -input_dir tables_dir -config_file configs/config.json -config_name pv_egy -feats "asp:p mod:i"
upload_iv_egy:
	python camel_morph/debugging/upload_sheets.py -input_dir tables_dir -config_file configs/config.json -config_name iv_egy -feats "asp:i mod:i"
upload_cv_egy:
	python camel_morph/debugging/upload_sheets.py -input_dir tables_dir -config_file configs/config.json -config_name cv_egy -feats "asp:c mod:i"

upload_adj_egy:
	python camel_morph/debugging/upload_sheets.py -input_dir tables_dir -config_file configs/config.json -config_name adj_egy_split_red -mode backup
upload_noun_egy:
	python camel_morph/debugging/upload_sheets.py -input_dir tables_dir -config_file configs/config.json -config_name noun_egy_split_red -mode backup
upload_noun_prop_egy:
	python camel_morph/debugging/upload_sheets.py -input_dir tables_dir -config_file configs/config.json -config_name noun_prop_egy_split_red -mode backup
upload_other_egy:
	python camel_morph/debugging/upload_sheets.py -input_dir tables_dir -config_file configs/config.json -config_name other_egy_split_red -mode backup

upload_nom_glf:
	python camel_morph/debugging/upload_sheets.py -input_dir tables_dir -config_file configs/config_capstones.json -config_name nom_glf_split_red -mode backup


msa_pv_process: download_msa_pv repr_lemmas_pv_msa make_db_pv_msa conj_pv_msa upload_pv_msa
msa_iv_i_process: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_i_msa upload_iv_i_msa
msa_iv_s_process: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_s_msa upload_iv_s_msa
msa_iv_j_process: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_j_msa upload_iv_j_msa
msa_iv_e_process: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_e_msa upload_iv_e_msa
msa_iv_x_process: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_x_msa upload_iv_x_msa
msa_cv_i_process: download_msa_cv repr_lemmas_cv_msa make_db_cv_msa conj_cv_i_msa upload_cv_i_msa
msa_cv_e_process: download_msa_cv repr_lemmas_cv_msa make_db_cv_msa conj_cv_e_msa upload_cv_e_msa
msa_cv_x_process: download_msa_cv repr_lemmas_cv_msa make_db_cv_msa conj_cv_x_msa upload_cv_x_msa

msa_adj_process: download_msa_adj repr_lemmas_adj_msa make_db_adj_msa conj_adj_msa upload_adj_msa
msa_noun_process: download_msa_noun repr_lemmas_noun_msa make_db_noun_msa conj_noun_msa upload_noun_msa
msa_noun_prop_process: download_msa_noun_prop repr_lemmas_noun_prop_msa make_db_noun_prop_msa conj_noun_prop_msa upload_noun_prop_msa
msa_other_process: download_msa_other repr_lemmas_other_msa make_db_other_msa conj_other_msa upload_other_msa

egy_adj_process: download_egy_adj repr_lemmas_adj_egy make_db_adj_egy conj_adj_egy upload_adj_egy
egy_noun_process: download_egy_noun repr_lemmas_noun_egy make_db_noun_egy conj_noun_egy upload_noun_egy
egy_noun_prop_process: download_egy_noun_prop repr_lemmas_noun_prop_egy make_db_noun_prop_egy conj_noun_prop_egy upload_noun_prop_egy
egy_other_process: download_egy_other repr_lemmas_other_egy make_db_other_egy conj_other_egy upload_other_egy

glf_nom_process: download_glf_nom make_db_nom_glf

glf_pv_process: download_glf_pv repr_lemmas_pv_glf make_db_pv_glf conj_pv_glf upload_pv_glf
glf_iv_process: download_glf_iv repr_lemmas_iv_glf make_db_iv_glf conj_iv_glf upload_iv_glf
glf_cv_process: download_glf_cv repr_lemmas_cv_glf make_db_cv_glf conj_cv_glf upload_cv_glf

egy_pv_process: download_egy_pv repr_lemmas_pv_egy make_db_pv_egy conj_pv_egy upload_pv_egy
egy_iv_process: download_egy_iv repr_lemmas_iv_egy make_db_iv_egy conj_iv_egy upload_iv_egy
egy_cv_process: download_egy_cv repr_lemmas_cv_egy make_db_cv_egy conj_cv_egy upload_cv_egy

msa_pv_bank_annotation:
	python camel_morph/debugging/paradigm_debugging.py -config_file configs/config.json -config_name pv_msa -feats "asp:p mod:i"
msa_pv_bank_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name pv_msa -feats "asp:p mod:i" -input_dir banks_dir -mode backup
msa_pv_auto_qc_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name pv_msa -feats "asp:p mod:i" -input_dir paradigm_debugging_dir -mode backup
msa_pv_debug: download_msa_pv repr_lemmas_pv_msa make_db_pv_msa conj_pv_msa msa_pv_bank_annotation msa_pv_bank_upload msa_pv_auto_qc_upload
msa_pv_debug_db_full: download_msa_all repr_lemmas_pv_msa make_db_all_aspects_msa conj_pv_msa_db_full msa_pv_bank_annotation msa_pv_bank_upload msa_pv_auto_qc_upload
msa_pv_debug_db_full_no_build: conj_pv_msa_db_full msa_pv_bank_annotation msa_pv_bank_upload msa_pv_auto_qc_upload

msa_iv_i_bank_annotation:
	python camel_morph/debugging/paradigm_debugging.py -config_file configs/config.json -config_name iv_msa -feats "asp:i mod:i"
msa_iv_i_bank_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name iv_msa -feats "asp:i mod:i" -input_dir banks_dir -mode backup
msa_iv_i_auto_qc_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name iv_msa -feats "asp:i mod:i" -input_dir paradigm_debugging_dir -mode backup
msa_iv_i_debug: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_i_msa msa_iv_i_bank_annotation msa_iv_i_bank_upload msa_iv_i_auto_qc_upload
msa_iv_i_debug_db_full: download_msa_all repr_lemmas_iv_msa make_db_all_aspects_msa conj_iv_i_msa_db_full msa_iv_i_bank_annotation msa_iv_i_bank_upload msa_iv_i_auto_qc_upload
msa_iv_i_debug_db_full_no_build: conj_iv_i_msa_db_full msa_iv_i_bank_annotation msa_iv_i_bank_upload msa_iv_i_auto_qc_upload

msa_iv_s_bank_annotation:
	python camel_morph/debugging/paradigm_debugging.py -config_file configs/config.json -config_name iv_msa -feats "asp:i mod:s"
msa_iv_s_bank_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name iv_msa -feats "asp:i mod:s" -input_dir banks_dir -mode backup
msa_iv_s_auto_qc_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name iv_msa -feats "asp:i mod:s" -input_dir paradigm_debugging_dir -mode backup
msa_iv_s_debug: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_s_msa msa_iv_s_bank_annotation msa_iv_s_bank_upload msa_iv_s_auto_qc_upload
msa_iv_s_debug_db_full: download_msa_all repr_lemmas_iv_msa make_db_all_aspects_msa conj_iv_s_msa_db_full msa_iv_s_bank_annotation msa_iv_s_bank_upload msa_iv_s_auto_qc_upload
msa_iv_s_debug_db_full_no_build: conj_iv_s_msa_db_full msa_iv_s_bank_annotation msa_iv_s_bank_upload msa_iv_s_auto_qc_upload

msa_iv_j_bank_annotation:
	python camel_morph/debugging/paradigm_debugging.py -config_file configs/config.json -config_name iv_msa -feats "asp:i mod:j"
msa_iv_j_bank_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name iv_msa -feats "asp:i mod:j" -input_dir banks_dir -mode backup
msa_iv_j_auto_qc_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name iv_msa -feats "asp:i mod:j" -input_dir paradigm_debugging_dir -mode backup
msa_iv_j_debug: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_j_msa msa_iv_j_bank_annotation msa_iv_j_bank_upload msa_iv_j_auto_qc_upload
msa_iv_j_debug_db_full: download_msa_all repr_lemmas_iv_msa make_db_all_aspects_msa conj_iv_j_msa_db_full msa_iv_j_bank_annotation msa_iv_j_bank_upload msa_iv_j_auto_qc_upload
msa_iv_j_debug_db_full_no_build: conj_iv_j_msa_db_full msa_iv_j_bank_annotation msa_iv_j_bank_upload msa_iv_j_auto_qc_upload

msa_iv_e_bank_annotation:
	python camel_morph/debugging/paradigm_debugging.py -config_file configs/config.json -config_name iv_msa -feats "asp:i mod:e"
msa_iv_e_bank_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name iv_msa -feats "asp:i mod:e" -input_dir banks_dir -mode backup
msa_iv_e_auto_qc_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name iv_msa -feats "asp:i mod:e" -input_dir paradigm_debugging_dir -mode backup
msa_iv_e_debug: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_e_msa msa_iv_e_bank_annotation msa_iv_e_bank_upload msa_iv_e_auto_qc_upload
msa_iv_e_debug_db_full: download_msa_all repr_lemmas_iv_msa make_db_all_aspects_msa conj_iv_e_msa_db_full msa_iv_e_bank_annotation msa_iv_e_bank_upload msa_iv_e_auto_qc_upload
msa_iv_e_debug_db_full_no_build: conj_iv_e_msa_db_full msa_iv_e_bank_annotation msa_iv_e_bank_upload msa_iv_e_auto_qc_upload

msa_iv_x_bank_annotation:
	python camel_morph/debugging/paradigm_debugging.py -config_file configs/config.json -config_name iv_msa -feats "asp:i mod:x" -process_key extra_energetic
msa_iv_x_bank_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name iv_msa -feats "asp:i mod:x" -input_dir banks_dir -mode backup
msa_iv_x_auto_qc_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name iv_msa -feats "asp:i mod:x" -input_dir paradigm_debugging_dir -mode backup
msa_iv_x_debug: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_x_msa msa_iv_x_bank_annotation msa_iv_x_bank_upload msa_iv_x_auto_qc_upload
msa_iv_x_debug_db_full: download_msa_all repr_lemmas_iv_msa make_db_all_aspects_msa conj_iv_x_msa_db_full msa_iv_x_bank_annotation msa_iv_x_bank_upload msa_iv_x_auto_qc_upload
msa_iv_x_debug_db_full_no_build: conj_iv_x_msa_db_full msa_iv_x_bank_annotation msa_iv_x_bank_upload msa_iv_x_auto_qc_upload

msa_cv_i_bank_annotation:
	python camel_morph/debugging/paradigm_debugging.py -config_file configs/config.json -config_name cv_msa -feats "asp:c mod:i"
msa_cv_i_bank_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name cv_msa -feats "asp:c mod:i" -input_dir banks_dir -mode backup
msa_cv_i_auto_qc_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name cv_msa -feats "asp:c mod:i" -input_dir paradigm_debugging_dir -mode backup
msa_cv_i_debug: download_msa_cv repr_lemmas_cv_msa make_db_cv_msa conj_cv_i_msa msa_cv_i_bank_annotation msa_cv_i_bank_upload msa_cv_i_auto_qc_upload
msa_cv_i_debug_db_full: download_msa_all repr_lemmas_cv_msa make_db_all_aspects_msa conj_cv_i_msa_db_full msa_cv_i_bank_annotation msa_cv_i_bank_upload msa_cv_i_auto_qc_upload
msa_cv_i_debug_db_full_no_build: conj_cv_i_msa_db_full msa_cv_i_bank_annotation msa_cv_i_bank_upload msa_cv_i_auto_qc_upload

msa_cv_e_bank_annotation:
	python camel_morph/debugging/paradigm_debugging.py -config_file configs/config.json -config_name cv_msa -feats "asp:c mod:e"
msa_cv_e_bank_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name cv_msa -feats "asp:c mod:e" -input_dir banks_dir -mode backup
msa_cv_e_auto_qc_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name cv_msa -feats "asp:c mod:e" -input_dir paradigm_debugging_dir -mode backup
msa_cv_e_debug: download_msa_cv repr_lemmas_cv_msa make_db_cv_msa conj_cv_e_msa msa_cv_e_bank_annotation msa_cv_e_bank_upload msa_cv_e_auto_qc_upload
msa_cv_e_debug_db_full: download_msa_all repr_lemmas_cv_msa make_db_all_aspects_msa conj_cv_e_msa_db_full msa_cv_e_bank_annotation msa_cv_e_bank_upload msa_cv_e_auto_qc_upload
msa_cv_e_debug_db_full_no_build: conj_cv_e_msa_db_full msa_cv_e_bank_annotation msa_cv_e_bank_upload msa_cv_e_auto_qc_upload

msa_cv_x_bank_annotation:
	python camel_morph/debugging/paradigm_debugging.py -config_file configs/config.json -config_name cv_msa -feats "asp:c mod:x" -process_key extra_energetic
msa_cv_x_bank_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name cv_msa -feats "asp:c mod:x" -input_dir banks_dir -mode backup
msa_cv_x_auto_qc_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name cv_msa -feats "asp:c mod:x" -input_dir paradigm_debugging_dir -mode backup
msa_cv_x_debug: download_msa_cv repr_lemmas_cv_msa make_db_cv_msa conj_cv_x_msa msa_cv_x_bank_annotation msa_cv_x_bank_upload msa_cv_x_auto_qc_upload
msa_cv_x_debug_db_full: download_msa_all repr_lemmas_cv_msa make_db_all_aspects_msa conj_cv_x_msa_db_full msa_cv_x_bank_annotation msa_cv_x_bank_upload msa_cv_x_auto_qc_upload
msa_cv_x_debug_db_full_no_build: conj_cv_x_msa_db_full msa_cv_x_bank_annotation msa_cv_x_bank_upload msa_cv_x_auto_qc_upload

msa_adj_bank_annotation:
	python camel_morph/debugging/paradigm_debugging.py -config_file configs/config.json -config_name adj_msa_split_red
msa_adj_bank_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name adj_msa_split_red -input_dir banks_dir -mode backup
msa_adj_auto_qc_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name adj_msa_split_red -input_dir paradigm_debugging_dir -mode backup
msa_adj_debug: download_msa_adj repr_lemmas_adj_msa make_db_adj_msa conj_adj_msa msa_adj_bank_annotation msa_adj_bank_upload msa_adj_auto_qc_upload

msa_noun_bank_annotation:
	python camel_morph/debugging/paradigm_debugging.py -config_file configs/config.json -config_name noun_msa_split_red
msa_noun_bank_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name noun_msa_split_red -input_dir banks_dir -mode backup
msa_noun_auto_qc_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name noun_msa_split_red -input_dir paradigm_debugging_dir -mode backup
msa_noun_debug: download_msa_noun repr_lemmas_noun_msa make_db_noun_msa conj_noun_msa msa_noun_bank_annotation msa_noun_bank_upload msa_noun_auto_qc_upload

msa_noun_prop_bank_annotation:
	python camel_morph/debugging/paradigm_debugging.py -config_file configs/config.json -config_name noun_prop_msa_split_red
msa_noun_prop_bank_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name noun_prop_msa_split_red -input_dir banks_dir -mode backup
msa_noun_prop_auto_qc_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name noun_prop_msa_split_red -input_dir paradigm_debugging_dir -mode backup
msa_noun_prop_debug: download_msa_noun_prop repr_lemmas_noun_prop_msa make_db_noun_prop_msa conj_noun_prop_msa msa_noun_prop_bank_annotation msa_noun_prop_bank_upload msa_noun_prop_auto_qc_upload

msa_other_bank_annotation:
	python camel_morph/debugging/paradigm_debugging.py -config_file configs/config.json -config_name other_msa_split_red
msa_other_bank_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name other_msa_split_red -input_dir banks_dir -mode backup
msa_other_auto_qc_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name other_msa_split_red -input_dir paradigm_debugging_dir -mode backup
msa_other_debug: download_msa_other repr_lemmas_other_msa make_db_other_msa conj_other_msa msa_other_bank_annotation msa_other_bank_upload msa_other_auto_qc_upload

egy_pv_bank_annotation:
	python camel_morph/debugging/paradigm_debugging.py -config_file configs/config.json -config_name pv_egy -feats "asp:p mod:i"
egy_pv_bank_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name pv_egy -feats "asp:p mod:i" -input_dir banks_dir -mode backup
egy_pv_auto_qc_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name pv_egy -feats "asp:p mod:i" -input_dir paradigm_debugging_dir -mode backup
egy_pv_debug: download_egy_pv repr_lemmas_pv_egy make_db_pv_egy conj_pv_egy egy_pv_bank_annotation egy_pv_bank_upload egy_pv_auto_qc_upload
egy_pv_debug_db_full: download_egy_all repr_lemmas_pv_egy make_db_all_aspects_egy conj_pv_egy_db_full egy_pv_bank_annotation egy_pv_bank_upload egy_pv_auto_qc_upload
egy_pv_debug_db_full_no_build: conj_pv_egy_db_full egy_pv_bank_annotation egy_pv_bank_upload egy_pv_auto_qc_upload

egy_iv_bank_annotation:
	python camel_morph/debugging/paradigm_debugging.py -config_file configs/config.json -config_name iv_egy -feats "asp:i mod:i"
egy_iv_bank_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name iv_egy -feats "asp:i mod:i" -input_dir banks_dir -mode backup
egy_iv_auto_qc_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name iv_egy -feats "asp:i mod:i" -input_dir paradigm_debugging_dir -mode backup
egy_iv_debug: download_egy_iv repr_lemmas_iv_egy make_db_iv_egy conj_iv_egy egy_iv_bank_annotation egy_iv_bank_upload egy_iv_auto_qc_upload
egy_iv_debug_db_full: download_egy_all repr_lemmas_iv_egy make_db_all_aspects_egy conj_iv_egy_db_full egy_iv_bank_annotation egy_iv_bank_upload egy_iv_auto_qc_upload
egy_iv_debug_db_full_no_build: conj_iv_egy_db_full egy_iv_bank_annotation egy_iv_bank_upload egy_iv_auto_qc_upload

egy_cv_bank_annotation:
	python camel_morph/debugging/paradigm_debugging.py -config_file configs/config.json -config_name cv_egy -feats "asp:c mod:i"
egy_cv_bank_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name cv_egy -feats "asp:c mod:i" -input_dir banks_dir -mode backup
egy_cv_auto_qc_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name cv_egy -feats "asp:c mod:i" -input_dir paradigm_debugging_dir -mode backup
egy_cv_debug: download_egy_cv repr_lemmas_cv_egy make_db_cv_egy conj_cv_egy egy_cv_bank_annotation egy_cv_bank_upload egy_cv_auto_qc_upload
egy_cv_debug_db_full: download_egy_all repr_lemmas_cv_egy make_db_all_aspects_egy conj_cv_egy_db_full egy_cv_bank_annotation egy_cv_bank_upload egy_cv_auto_qc_upload
egy_cv_debug_db_full_no_build: conj_cv_egy_db_full egy_cv_bank_annotation egy_cv_bank_upload egy_cv_auto_qc_upload

egy_adj_bank_annotation:
	python camel_morph/debugging/paradigm_debugging.py -config_file configs/config.json -config_name adj_egy_split_red
egy_adj_bank_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name adj_egy_split_red -input_dir banks_dir -mode backup
egy_adj_auto_qc_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name adj_egy_split_red -input_dir paradigm_debugging_dir -mode backup
egy_adj_debug: download_egy_adj repr_lemmas_adj_egy make_db_adj_egy conj_adj_egy egy_adj_bank_annotation egy_adj_bank_upload egy_adj_auto_qc_upload

egy_noun_bank_annotation:
	python camel_morph/debugging/paradigm_debugging.py -config_file configs/config.json -config_name noun_egy_split_red
egy_noun_bank_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name noun_egy_split_red -input_dir banks_dir -mode backup
egy_noun_auto_qc_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config.json -config_name noun_egy_split_red -input_dir paradigm_debugging_dir -mode backup
egy_noun_debug: download_egy_noun repr_lemmas_noun_egy make_db_noun_egy conj_noun_egy egy_noun_bank_annotation egy_noun_bank_upload egy_noun_auto_qc_upload

eval_camel_tb_compare:
	python eval/evaluate_camel_morph.py -data_path eval_files/camel_tb_uniq_types.txt -preprocessing camel_tb -db_dir databases -config_file configs/config.json -config_file configs/config.json -config_name all_aspects_msa -camel_tools $(camel_tools) -baseline_db eval_files/calima-msa-s31_0.4.2.utf8.db -eval_mode compare -results_path eval_files/camel_tb_compare.tsv -n 100000

msa_all_debug_db_full: msa_pv_debug_db_full msa_iv_i_debug_db_full msa_iv_s_debug_db_full msa_iv_j_debug_db_full msa_iv_e_debug_db_full msa_iv_x_debug_db_full msa_cv_i_debug_db_full msa_cv_e_debug_db_full msa_cv_x_debug_db_full
msa_all_debug_db_full_build: download_msa_all repr_lemmas_pv_msa repr_lemmas_iv_msa repr_lemmas_cv_msa make_db_all_aspects_msa
msa_all_debug_db_full_no_build: msa_pv_debug_db_full_no_build msa_iv_i_debug_db_full_no_build msa_iv_s_debug_db_full_no_build msa_iv_j_debug_db_full_no_build msa_iv_e_debug_db_full_no_build msa_iv_x_debug_db_full_no_build msa_cv_i_debug_db_full_no_build msa_cv_e_debug_db_full_no_build msa_cv_x_debug_db_full_no_build

egy_all_debug_db_full: egy_pv_debug_db_full egy_iv_debug_db_full egy_cv_debug_db_full
egy_all_debug_db_full_build: download_egy_all repr_lemmas_pv_egy repr_lemmas_iv_egy repr_lemmas_cv_egy make_db_all_aspects_egy
egy_all_debug_db_full_no_build: egy_pv_debug_db_full_no_build egy_iv_debug_db_full_no_build egy_cv_debug_db_full_no_build

eval_camel_tb_backoff:
	python eval/evaluate_camel_morph.py -data_path eval_files/ATB123-train.102312.calima-msa-s31_0.3.0.magold -preprocessing magold -eval_mode recall_backoff -results_path eval_files/ATB-Recall-Backoff.tsv -n 100000 -db_dir databases -config_file configs/config.json -config_file configs/config.json -config_name all_aspects_msa -camel_tools $(camel_tools) -baseline_db eval_files/calima-egy-c044_0.2.0.utf8.db

eval_arz_recall:
	python eval/evaluate_camel_morph.py -data_path eval_files/ARZ-All-train.113012.magold -preprocessing magold -eval_mode recall -results_path eval_files/ARZ-Recall-v5.tsv -n 100000 -db_dir databases -config_file configs/config.json -config_file configs/config.json -config_name all_aspects_egy -camel_tools $(camel_tools) -baseline_db eval_files/calima-egy-c044_0.2.0.utf8.db -msa_analysis_union_config_name all_aspects_msa -input_source ldc_dediac_processed

download_egy_nom_docs:	
	python camel_morph/debugging/download_sheets.py -config_file configs/config_docs.json -config_name nominals_egy_docs
make_db_egy_nom_docs:
	python camel_morph/db_maker.py -config_file configs/config_docs.json -config_name nominals_egy_docs
generate_auto_docs_egy_nom:
	python camel_morph/debugging/generate_docs_tables.py -config_file configs/config_docs.json -config_name nominals_egy_docs -most_prob_lemma
upload_auto_docs_egy_nom:
	python camel_morph/debugging/upload_sheets.py -input_dir docs_tables_dir -config_file configs/config_docs.json -config_name nominals_egy_docs -mode backup -insert_index
auto_docs_egy_nom_process: download_egy_nom_docs make_db_egy_nom_docs generate_auto_docs_egy_nom upload_auto_docs_egy_nom

egy_nom_docs_bank_annotation:
	python camel_morph/debugging/automatic_docs_debugging.py -config_file configs/config_docs.json -config_name nominals_egy_docs
egy_nom_docs_bank_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config_docs.json -config_name nominals_egy_docs -input_dir docs_banks_dir -mode backup
egy_nom_docs_auto_qc_upload:
	python camel_morph/debugging/upload_sheets.py -config_file configs/config_docs.json -config_name nominals_egy_docs -input_dir docs_debugging_dir -mode backup -insert_index
egy_auto_docs_debug: download_egy_nom_docs make_db_egy_nom_docs generate_auto_docs_egy_nom egy_nom_docs_bank_annotation egy_nom_docs_bank_upload egy_nom_docs_auto_qc_upload

download_glf_nom_docs:	
	python camel_morph/debugging/download_sheets.py -config_file configs/config_capstones.json -config_name nominals_glf_docs
make_db_glf_nom_docs:
	python camel_morph/db_maker.py -config_file configs/config_capstones.json -config_name nominals_glf_docs
generate_auto_docs_glf_nom:
	python camel_morph/debugging/generate_docs_tables.py -config_file configs/config_capstones.json -config_name nominals_glf_docs -most_prob_lemma
upload_auto_docs_glf_nom:
	python camel_morph/debugging/upload_sheets.py -input_dir docs_tables_dir -config_file configs/config_capstones.json -config_name nominals_glf_docs -mode backup -insert_index
auto_docs_glf_nom_process: download_glf_nom_docs make_db_glf_nom_docs generate_auto_docs_glf_nom upload_auto_docs_glf_nom

glf_pilot:
	python camel_morph/sandbox/glf_pilot.py -config_file_egy configs/config_docs.json -config_name_egy nominals_egy_docs -config_file_glf configs/config_capstones.json -config_name_glf nom_glf_split_red -config_name_glf_docs nominals_glf_docs -eval_mode recall_glf_magold_raw_no_lex
glf_pilot_mp:
	python camel_morph/sandbox/disambig_experiment.py

debug_eval_gen:
	python camel_morph/eval/debug_evaluate_full_generative.py -msa_config_name all_aspects_msa -n 2 -camel_tools local -db_dir databases/camel-morph-msa -feats_debug mod:u\>i mod:u\>s mod:u\>j gen:u prc0:0 prc1:0 enc0:0 -feat_combs 2 -report_dir eval_files/results_multiproc_all_2

evab_gen_verb:
	python camel_morph/eval/evaluate_full_generative.py -db_system databases/camel-morph-msa/XYZ_msa_all_v1.0.db -camel_tools local -report_dir eval_files/results_multiproc_all_1 -possible_feat_combs eval_files/results_multiproc_all_1/possible_feat_combs.pkl -multiprocessing -pos verb

evab_gen_noun:
	python camel_morph/eval/evaluate_full_generative.py -db_system databases/camel-morph-msa/XYZ_msa_nom_v1.0.db -camel_tools local -report_dir eval_files/results_multiproc_20_noun -pos noun -multiprocessing