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

make_db_nom_msa:	
	python camel_morph/db_maker.py -config_file configs/config.json -config_name nom_msa_red 

repr_lemmas_pv_msa:
	python camel_morph/debugging/create_repr_lemmas_list.py -config_file configs/config.json -config_name pv_msa -display_format expanded -feats "asp:p mod:i" -db eval_files/calima-msa-s31_0.4.2.utf8.db
repr_lemmas_iv_msa:	
	python camel_morph/debugging/create_repr_lemmas_list.py -config_file configs/config.json -config_name iv_msa -display_format expanded -feats "asp:i mod:i" -db eval_files/calima-msa-s31_0.4.2.utf8.db
repr_lemmas_cv_msa:	
	python camel_morph/debugging/create_repr_lemmas_list.py -config_file configs/config.json -config_name cv_msa -display_format expanded -feats "asp:c mod:i" -db eval_files/calima-msa-s31_0.4.2.utf8.db
repr_lemmas_pv_glf:	
	python camel_morph/debugging/create_repr_lemmas_list.py -config_file configs/config.json -config_name pv_glf_red -display_format expanded -feats "asp:p mod:i"
repr_lemmas_iv_glf:	
	python camel_morph/debugging/create_repr_lemmas_list.py -config_file configs/config.json -config_name iv_glf_red -display_format expanded -feats "asp:i mod:i"
repr_lemmas_cv_glf:	
	python camel_morph/debugging/create_repr_lemmas_list.py -config_file configs/config.json -config_name cv_glf_red -display_format expanded -feats "asp:c mod:i"
repr_lemmas_pv_egy:	
	python camel_morph/debugging/create_repr_lemmas_list.py -config_file configs/config.json -config_name pv_egy -display_format expanded -feats "asp:p mod:i"
repr_lemmas_iv_egy:	
	python camel_morph/debugging/create_repr_lemmas_list.py -config_file configs/config.json -config_name iv_egy -display_format expanded -feats "asp:i mod:i"
repr_lemmas_cv_egy:	
	python camel_morph/debugging/create_repr_lemmas_list.py -config_file configs/config.json -config_name cv_egy -display_format expanded -feats "asp:c mod:i"
repr_lemmas_nom_msa:	
	python camel_morph/debugging/create_repr_lemmas_list.py -config_file configs/config.json -config_name nom_msa_split -display_format expanded -db eval_files/calima-msa-s31_0.4.2.utf8.db
repr_lemmas_adj_msa:
	python camel_morph/debugging/create_repr_lemmas_list.py -config_file configs/config.json -config_name adj_msa_split_red -display_format expanded -db eval_files/calima-msa-s31_0.4.2.utf8.db
repr_lemmas_noun_msa:
	python camel_morph/debugging/create_repr_lemmas_list.py -config_file configs/config.json -config_name noun_msa_split_red adj_msa_split_red -display_format expanded -db eval_files/calima-msa-s31_0.4.2.utf8.db

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
	python camel_morph/debugging/generate_conj_table.py -feats "asp:p mod:i" -config_file configs/config.json -config_name pv_msa -db XYZ_msa_cr_v1.0.db

conj_iv_i_msa_db_full:	
	python camel_morph/debugging/generate_conj_table.py -feats "asp:i mod:i" -config_file configs/config.json -config_name iv_msa -db XYZ_msa_cr_v1.0.db
conj_iv_s_msa_db_full:	
	python camel_morph/debugging/generate_conj_table.py -feats "asp:i mod:s" -config_file configs/config.json -config_name iv_msa -db XYZ_msa_cr_v1.0.db
conj_iv_j_msa_db_full:	
	python camel_morph/debugging/generate_conj_table.py -feats "asp:i mod:j" -config_file configs/config.json -config_name iv_msa -db XYZ_msa_cr_v1.0.db
conj_iv_e_msa_db_full:	
	python camel_morph/debugging/generate_conj_table.py -feats "asp:i mod:e" -config_file configs/config.json -config_name iv_msa -db XYZ_msa_cr_v1.0.db
conj_iv_x_msa_db_full:	
	python camel_morph/debugging/generate_conj_table.py -feats "asp:i mod:x" -config_file configs/config.json -config_name iv_msa -db XYZ_msa_cr_v1.0.db

conj_cv_i_msa_db_full:	
	python camel_morph/debugging/generate_conj_table.py -feats "asp:c mod:i" -config_file configs/config.json -config_name cv_msa -db XYZ_msa_cr_v1.0.db
conj_cv_e_msa_db_full:	
	python camel_morph/debugging/generate_conj_table.py -feats "asp:c mod:e" -config_file configs/config.json -config_name cv_msa -db XYZ_msa_cr_v1.0.db
conj_cv_x_msa_db_full:	
	python camel_morph/debugging/generate_conj_table.py -feats "asp:c mod:x" -config_file configs/config.json -config_name cv_msa -db XYZ_msa_cr_v1.0.db

conj_adj_msa:	
	python camel_morph/debugging/generate_conj_table.py -config_file configs/config.json -config_name adj_msa_split_red
conj_noun_msa:	
	python camel_morph/debugging/generate_conj_table.py -config_file configs/config.json -config_name noun_msa_split_red

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

download_msa_nom:	
	python camel_morph/debugging/download_sheets.py -config_file configs/config.json -config_name nom_msa_red

download_egy_nom:	
	python camel_morph/debugging/download_sheets.py -config_file configs/config.json -config_name nom_egy_red

upload_pv_msa:
	python camel_morph/debugging/upload_sheets.py -formatting conj_tables -config_file configs/config.json -config_name pv_msa -feats "asp:p mod:i"

upload_iv_i_msa:
	python camel_morph/debugging/upload_sheets.py -formatting conj_tables -config_file configs/config.json -config_name iv_msa -feats "asp:i mod:i"
upload_iv_s_msa:
	python camel_morph/debugging/upload_sheets.py -formatting conj_tables -config_file configs/config.json -config_name iv_msa -feats "asp:i mod:s"
upload_iv_j_msa:
	python camel_morph/debugging/upload_sheets.py -formatting conj_tables -config_file configs/config.json -config_name iv_msa -feats "asp:i mod:j"
upload_iv_e_msa:
	python camel_morph/debugging/upload_sheets.py -formatting conj_tables -config_file configs/config.json -config_name iv_msa -feats "asp:i mod:e"
upload_iv_x_msa:
	python camel_morph/debugging/upload_sheets.py -formatting conj_tables -config_file configs/config.json -config_name iv_msa -feats "asp:i mod:x"

upload_cv_i_msa:
	python camel_morph/debugging/upload_sheets.py -formatting conj_tables -config_file configs/config.json -config_name cv_msa -feats "asp:c mod:i"
upload_cv_e_msa:
	python camel_morph/debugging/upload_sheets.py -formatting conj_tables -config_file configs/config.json -config_name cv_msa -feats "asp:c mod:e"
upload_cv_x_msa:
	python camel_morph/debugging/upload_sheets.py -formatting conj_tables -config_file configs/config.json -config_name cv_msa -feats "asp:c mod:x"

upload_adj_msa:
	python camel_morph/debugging/upload_sheets.py -formatting conj_tables -config_file configs/config.json -config_name adj_msa_split_red -mode backup
upload_noun_msa:
	python camel_morph/debugging/upload_sheets.py -formatting conj_tables -config_file configs/config.json -config_name noun_msa_split_red -mode backup

upload_pv_glf:
	python camel_morph/debugging/upload_sheets.py -formatting conj_tables -config_file configs/config.json -config_name pv_glf_red -feats "asp:p mod:i" -mode backup
upload_iv_glf:
	python camel_morph/debugging/upload_sheets.py -formatting conj_tables -config_file configs/config.json -config_name iv_glf_red -feats "asp:i mod:i" -mode backup
upload_cv_glf:
	python camel_morph/debugging/upload_sheets.py -formatting conj_tables -config_file configs/config.json -config_name cv_glf_red -feats "asp:c mod:i" -mode backup
upload_pv_egy:
	python camel_morph/debugging/upload_sheets.py -formatting conj_tables -config_file configs/config.json -config_name pv_egy -feats "asp:p mod:i"
upload_iv_egy:
	python camel_morph/debugging/upload_sheets.py -formatting conj_tables -config_file configs/config.json -config_name iv_egy -feats "asp:i mod:i"
upload_cv_egy:
	python camel_morph/debugging/upload_sheets.py -formatting conj_tables -config_file configs/config.json -config_name cv_egy -feats "asp:c mod:i"

msa_pv_process: download_msa_pv repr_lemmas_pv_msa make_db_pv_msa conj_pv_msa upload_pv_msa
msa_iv_i_process: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_i_msa upload_iv_i_msa
msa_iv_s_process: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_s_msa upload_iv_s_msa
msa_iv_j_process: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_j_msa upload_iv_j_msa
msa_iv_e_process: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_e_msa upload_iv_e_msa
msa_iv_x_process: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_x_msa upload_iv_x_msa
msa_cv_i_process: download_msa_cv repr_lemmas_cv_msa make_db_cv_msa conj_cv_i_msa upload_cv_i_msa
msa_cv_e_process: download_msa_cv repr_lemmas_cv_msa make_db_cv_msa conj_cv_e_msa upload_cv_e_msa
msa_cv_x_process: download_msa_cv repr_lemmas_cv_msa make_db_cv_msa conj_cv_x_msa upload_cv_x_msa

msa_adj_process: download_msa_nom repr_lemmas_adj_msa make_db_nom_msa conj_adj_msa upload_adj_msa
msa_noun_process: download_msa_nom repr_lemmas_noun_msa make_db_nom_msa conj_noun_msa upload_noun_msa

glf_pv_process: download_glf_pv repr_lemmas_pv_glf make_db_pv_glf conj_pv_glf upload_pv_glf
glf_iv_process: download_glf_iv repr_lemmas_iv_glf make_db_iv_glf conj_iv_glf upload_iv_glf
glf_cv_process: download_glf_cv repr_lemmas_cv_glf make_db_cv_glf conj_cv_glf upload_cv_glf

egy_pv_process: download_egy_pv repr_lemmas_pv_egy make_db_pv_egy conj_pv_egy upload_pv_egy
egy_iv_process: download_egy_iv repr_lemmas_iv_egy make_db_iv_egy conj_iv_egy upload_iv_egy
egy_cv_process: download_egy_cv repr_lemmas_cv_egy make_db_cv_egy conj_cv_egy upload_cv_egy

msa_pv_bank_annotation:
	python paradigm_debugging.py -config_file configs/config.json -config_name pv_msa -feats "asp:p mod:i"
msa_pv_bank_upload:
	python camel_morph/debugging/upload_sheets.py -dir camel_morph/debugging/banks -file_name MSA-PV-Bank.tsv -spreadsheet_name Paradigm-Banks -gsheet_name MSA-PV-Bank -formatting bank -mode backup
msa_pv_auto_qc_upload:
	python camel_morph/debugging/upload_sheets.py -dir camel_morph/debugging/paradigm_debugging -file_name paradigm_debug_pv_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-PV -formatting conj_tables -mode backup
msa_pv_debug: download_msa_pv repr_lemmas_pv_msa make_db_pv_msa conj_pv_msa msa_pv_bank_annotation msa_pv_bank_upload msa_pv_auto_qc_upload
msa_pv_debug_db_full: download_msa_all repr_lemmas_pv_msa make_db_all_aspects_msa conj_pv_msa_db_full msa_pv_bank_annotation msa_pv_bank_upload msa_pv_auto_qc_upload
msa_pv_debug_db_full_no_build: conj_pv_msa_db_full msa_pv_bank_annotation msa_pv_bank_upload msa_pv_auto_qc_upload

msa_iv_i_bank_annotation:
	python paradigm_debugging.py -config_file configs/config.json -config_name iv_msa -feats "asp:i mod:i"
msa_iv_i_bank_upload:
	python camel_morph/debugging/upload_sheets.py -dir camel_morph/debugging/banks -file_name MSA-IV-Ind-Bank.tsv -spreadsheet_name Paradigm-Banks -gsheet_name MSA-IV-Ind-Bank -formatting bank -mode backup
msa_iv_i_auto_qc_upload:
	python camel_morph/debugging/upload_sheets.py -dir camel_morph/debugging/paradigm_debugging -file_name paradigm_debug_iv-i_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-IV-Ind -formatting conj_tables -mode backup
msa_iv_i_debug: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_i_msa msa_iv_i_bank_annotation msa_iv_i_bank_upload msa_iv_i_auto_qc_upload
msa_iv_i_debug_db_full: download_msa_all repr_lemmas_iv_msa make_db_all_aspects_msa conj_iv_i_msa_db_full msa_iv_i_bank_annotation msa_iv_i_bank_upload msa_iv_i_auto_qc_upload
msa_iv_i_debug_db_full_no_build: conj_iv_i_msa_db_full msa_iv_i_bank_annotation msa_iv_i_bank_upload msa_iv_i_auto_qc_upload

msa_iv_s_bank_annotation:
	python paradigm_debugging.py -config_file configs/config.json -config_name iv_msa -feats "asp:i mod:s"
msa_iv_s_bank_upload:
	python camel_morph/debugging/upload_sheets.py -dir camel_morph/debugging/banks -file_name MSA-IV-Sub-Bank.tsv -spreadsheet_name Paradigm-Banks -gsheet_name MSA-IV-Sub-Bank -formatting bank -mode backup
msa_iv_s_auto_qc_upload:
	python camel_morph/debugging/upload_sheets.py -dir camel_morph/debugging/paradigm_debugging -file_name paradigm_debug_iv-s_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-IV-Sub -formatting conj_tables -mode backup
msa_iv_s_debug: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_s_msa msa_iv_s_bank_annotation msa_iv_s_bank_upload msa_iv_s_auto_qc_upload
msa_iv_s_debug_db_full: download_msa_all repr_lemmas_iv_msa make_db_all_aspects_msa conj_iv_s_msa_db_full msa_iv_s_bank_annotation msa_iv_s_bank_upload msa_iv_s_auto_qc_upload
msa_iv_s_debug_db_full_no_build: conj_iv_s_msa_db_full msa_iv_s_bank_annotation msa_iv_s_bank_upload msa_iv_s_auto_qc_upload

msa_iv_j_bank_annotation:
	python paradigm_debugging.py -config_file configs/config.json -config_name iv_msa -feats "asp:i mod:j"
msa_iv_j_bank_upload:
	python camel_morph/debugging/upload_sheets.py -dir camel_morph/debugging/banks -file_name MSA-IV-Jus-Bank.tsv -spreadsheet_name Paradigm-Banks -gsheet_name MSA-IV-Jus-Bank -formatting bank -mode backup
msa_iv_j_auto_qc_upload:
	python camel_morph/debugging/upload_sheets.py -dir camel_morph/debugging/paradigm_debugging -file_name paradigm_debug_iv-j_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-IV-Jus -formatting conj_tables -mode backup
msa_iv_j_debug: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_j_msa msa_iv_j_bank_annotation msa_iv_j_bank_upload msa_iv_j_auto_qc_upload
msa_iv_j_debug_db_full: download_msa_all repr_lemmas_iv_msa make_db_all_aspects_msa conj_iv_j_msa_db_full msa_iv_j_bank_annotation msa_iv_j_bank_upload msa_iv_j_auto_qc_upload
msa_iv_j_debug_db_full_no_build: conj_iv_j_msa_db_full msa_iv_j_bank_annotation msa_iv_j_bank_upload msa_iv_j_auto_qc_upload

msa_iv_e_bank_annotation:
	python paradigm_debugging.py -config_file configs/config.json -config_name iv_msa -feats "asp:i mod:e"
msa_iv_e_bank_upload:
	python camel_morph/debugging/upload_sheets.py -dir camel_morph/debugging/banks -file_name "MSA-IV-(X)Ener-Bank.tsv" -spreadsheet_name Paradigm-Banks -gsheet_name "MSA-IV-(X)Ener-Bank" -formatting bank -mode backup
msa_iv_e_auto_qc_upload:
	python camel_morph/debugging/upload_sheets.py -dir camel_morph/debugging/paradigm_debugging -file_name paradigm_debug_iv-e_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-IV-Ener -formatting conj_tables -mode backup
msa_iv_e_debug: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_e_msa msa_iv_e_bank_annotation msa_iv_e_bank_upload msa_iv_e_auto_qc_upload
msa_iv_e_debug_db_full: download_msa_all repr_lemmas_iv_msa make_db_all_aspects_msa conj_iv_e_msa_db_full msa_iv_e_bank_annotation msa_iv_e_bank_upload msa_iv_e_auto_qc_upload
msa_iv_e_debug_db_full_no_build: conj_iv_e_msa_db_full msa_iv_e_bank_annotation msa_iv_e_bank_upload msa_iv_e_auto_qc_upload

msa_iv_x_bank_annotation:
	python paradigm_debugging.py -config_file configs/config.json -config_name iv_msa -feats "asp:i mod:x" -process_key extra_energetic
msa_iv_x_bank_upload:
	python camel_morph/debugging/upload_sheets.py -dir camel_morph/debugging/banks -file_name "MSA-IV-(X)Ener-Bank.tsv" -spreadsheet_name Paradigm-Banks -gsheet_name "MSA-IV-(X)Ener-Bank" -formatting bank -mode backup
msa_iv_x_auto_qc_upload:
	python camel_morph/debugging/upload_sheets.py -dir camel_morph/debugging/paradigm_debugging -file_name paradigm_debug_iv-x_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-IV-XEner -formatting conj_tables -mode backup
msa_iv_x_debug: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_x_msa msa_iv_x_bank_annotation msa_iv_x_bank_upload msa_iv_x_auto_qc_upload
msa_iv_x_debug_db_full: download_msa_all repr_lemmas_iv_msa make_db_all_aspects_msa conj_iv_x_msa_db_full msa_iv_x_bank_annotation msa_iv_x_bank_upload msa_iv_x_auto_qc_upload
msa_iv_x_debug_db_full_no_build: conj_iv_x_msa_db_full msa_iv_x_bank_annotation msa_iv_x_bank_upload msa_iv_x_auto_qc_upload

msa_cv_i_bank_annotation:
	python paradigm_debugging.py -config_file configs/config.json -config_name cv_msa -feats "asp:c mod:i"
msa_cv_i_bank_upload:
	python camel_morph/debugging/upload_sheets.py -dir camel_morph/debugging/banks -file_name MSA-CV-Ind-Bank.tsv -spreadsheet_name Paradigm-Banks -gsheet_name MSA-CV-Ind-Bank -formatting bank -mode backup
msa_cv_i_auto_qc_upload:
	python camel_morph/debugging/upload_sheets.py -dir camel_morph/debugging/paradigm_debugging -file_name paradigm_debug_cv-i_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-CV-Ind -formatting conj_tables -mode backup
msa_cv_i_debug: download_msa_cv repr_lemmas_cv_msa make_db_cv_msa conj_cv_i_msa msa_cv_i_bank_annotation msa_cv_i_bank_upload msa_cv_i_auto_qc_upload
msa_cv_i_debug_db_full: download_msa_all repr_lemmas_cv_msa make_db_all_aspects_msa conj_cv_i_msa_db_full msa_cv_i_bank_annotation msa_cv_i_bank_upload msa_cv_i_auto_qc_upload
msa_cv_i_debug_db_full_no_build: conj_cv_i_msa_db_full msa_cv_i_bank_annotation msa_cv_i_bank_upload msa_cv_i_auto_qc_upload

msa_cv_e_bank_annotation:
	python paradigm_debugging.py -config_file configs/config.json -config_name cv_msa -feats "asp:c mod:e"
msa_cv_e_bank_upload:
	python camel_morph/debugging/upload_sheets.py -dir camel_morph/debugging/banks -file_name "MSA-CV-(X)Ener-Bank.tsv" -spreadsheet_name Paradigm-Banks -gsheet_name "MSA-CV-(X)Ener-Bank" -formatting bank -mode backup
msa_cv_e_auto_qc_upload:
	python camel_morph/debugging/upload_sheets.py -dir camel_morph/debugging/paradigm_debugging -file_name paradigm_debug_cv-e_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-CV-Ener -formatting conj_tables -mode backup
msa_cv_e_debug: download_msa_cv repr_lemmas_cv_msa make_db_cv_msa conj_cv_e_msa msa_cv_e_bank_annotation msa_cv_e_bank_upload msa_cv_e_auto_qc_upload
msa_cv_e_debug_db_full: download_msa_all repr_lemmas_cv_msa make_db_all_aspects_msa conj_cv_e_msa_db_full msa_cv_e_bank_annotation msa_cv_e_bank_upload msa_cv_e_auto_qc_upload
msa_cv_e_debug_db_full_no_build: conj_cv_e_msa_db_full msa_cv_e_bank_annotation msa_cv_e_bank_upload msa_cv_e_auto_qc_upload

msa_cv_x_bank_annotation:
	python paradigm_debugging.py -config_file configs/config.json -config_name cv_msa -feats "asp:c mod:x" -process_key extra_energetic
msa_cv_x_bank_upload:
	python camel_morph/debugging/upload_sheets.py -dir camel_morph/debugging/banks -file_name "MSA-CV-(X)Ener-Bank.tsv" -spreadsheet_name Paradigm-Banks -gsheet_name "MSA-CV-(X)Ener-Bank" -formatting bank -mode backup
msa_cv_x_auto_qc_upload:
	python camel_morph/debugging/upload_sheets.py -dir camel_morph/debugging/paradigm_debugging -file_name paradigm_debug_cv-x_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-CV-XEner -formatting conj_tables -mode backup
msa_cv_x_debug: download_msa_cv repr_lemmas_cv_msa make_db_cv_msa conj_cv_x_msa msa_cv_x_bank_annotation msa_cv_x_bank_upload msa_cv_x_auto_qc_upload
msa_cv_x_debug_db_full: download_msa_all repr_lemmas_cv_msa make_db_all_aspects_msa conj_cv_x_msa_db_full msa_cv_x_bank_annotation msa_cv_x_bank_upload msa_cv_x_auto_qc_upload
msa_cv_x_debug_db_full_no_build: conj_cv_x_msa_db_full msa_cv_x_bank_annotation msa_cv_x_bank_upload msa_cv_x_auto_qc_upload

msa_adj_bank_annotation:
	python paradigm_debugging.py -config_file configs/config.json -config_name adj_msa_split_red
msa_adj_bank_upload:
	python camel_morph/debugging/upload_sheets.py -dir camel_morph/debugging/banks -file_name MSA-Adj-Bank.tsv -spreadsheet_name Paradigm-Banks -gsheet_name MSA-Adj-Bank -formatting bank -mode backup
msa_adj_auto_qc_upload:
	python camel_morph/debugging/upload_sheets.py -dir camel_morph/debugging/paradigm_debugging -file_name paradigm_debug_adj_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging-Nominals -gsheet_name MSA-Adj -formatting conj_tables -mode backup
msa_adj_debug: download_msa_nom repr_lemmas_adj_msa make_db_nom_msa conj_adj_msa msa_adj_bank_annotation msa_adj_bank_upload msa_adj_auto_qc_upload
msa_adj_debug_db_full: download_msa_nom repr_lemmas_adj_msa make_db_nom_msa conj_adj_msa_db_full msa_adj_bank_annotation msa_adj_bank_upload msa_adj_auto_qc_upload
msa_adj_debug_db_full_no_build: conj_adj_msa_db_full msa_adj_bank_annotation msa_adj_bank_upload msa_adj_auto_qc_upload

egy_pv_bank_annotation:
	python paradigm_debugging.py -config_file configs/config.json -config_name pv_egy -feats "asp:p mod:i"
egy_pv_bank_upload:
	python camel_morph/debugging/upload_sheets.py -dir camel_morph/debugging/banks -file_name EGY-PV-Bank.tsv -spreadsheet_name Paradigm-Banks -gsheet_name EGY-PV-Bank -formatting bank -mode backup
egy_pv_auto_qc_upload:
	python camel_morph/debugging/upload_sheets.py -dir camel_morph/debugging/paradigm_debugging -file_name paradigm_debug_pv_egy_v1.0.tsv -spreadsheet_name Paradigm-Debugging-Dialects -gsheet_name EGY-PV -formatting conj_tables -mode backup
egy_pv_debug: download_egy_pv repr_lemmas_pv_egy make_db_pv_egy conj_pv_egy egy_pv_bank_annotation egy_pv_bank_upload egy_pv_auto_qc_upload
egy_pv_debug_db_full: download_egy_all repr_lemmas_pv_egy make_db_all_aspects_egy conj_pv_egy_db_full egy_pv_bank_annotation egy_pv_bank_upload egy_pv_auto_qc_upload
egy_pv_debug_db_full_no_build: conj_pv_egy_db_full egy_pv_bank_annotation egy_pv_bank_upload egy_pv_auto_qc_upload

egy_iv_bank_annotation:
	python paradigm_debugging.py -config_file configs/config.json -config_name iv_egy -feats "asp:i mod:i"
egy_iv_bank_upload:
	python camel_morph/debugging/upload_sheets.py -dir camel_morph/debugging/banks -file_name EGY-IV-Bank.tsv -spreadsheet_name Paradigm-Banks -gsheet_name EGY-IV-Bank -formatting bank -mode backup
egy_iv_auto_qc_upload:
	python camel_morph/debugging/upload_sheets.py -dir camel_morph/debugging/paradigm_debugging -file_name paradigm_debug_iv_egy_v1.0.tsv -spreadsheet_name Paradigm-Debugging-Dialects -gsheet_name EGY-IV -formatting conj_tables -mode backup
egy_iv_debug: download_egy_iv repr_lemmas_iv_egy make_db_iv_egy conj_iv_egy egy_iv_bank_annotation egy_iv_bank_upload egy_iv_auto_qc_upload
egy_iv_debug_db_full: download_egy_all repr_lemmas_iv_egy make_db_all_aspects_egy conj_iv_egy_db_full egy_iv_bank_annotation egy_iv_bank_upload egy_iv_auto_qc_upload
egy_iv_debug_db_full_no_build: conj_iv_egy_db_full egy_iv_bank_annotation egy_iv_bank_upload egy_iv_auto_qc_upload

egy_cv_bank_annotation:
	python paradigm_debugging.py -config_file configs/config.json -config_name cv_egy -feats "asp:c mod:i"
egy_cv_bank_upload:
	python camel_morph/debugging/upload_sheets.py -dir camel_morph/debugging/banks -file_name EGY-CV-Bank.tsv -spreadsheet_name Paradigm-Banks -gsheet_name EGY-CV-Bank -formatting bank -mode backup
egy_cv_auto_qc_upload:
	python camel_morph/debugging/upload_sheets.py -dir camel_morph/debugging/paradigm_debugging -file_name paradigm_debug_cv_egy_v1.0.tsv -spreadsheet_name Paradigm-Debugging-Dialects -gsheet_name EGY-CV -formatting conj_tables -mode backup
egy_cv_debug: download_egy_cv repr_lemmas_cv_egy make_db_cv_egy conj_cv_egy egy_cv_bank_annotation egy_cv_bank_upload egy_cv_auto_qc_upload
egy_cv_debug_db_full: download_egy_all repr_lemmas_cv_egy make_db_all_aspects_egy conj_cv_egy_db_full egy_cv_bank_annotation egy_cv_bank_upload egy_cv_auto_qc_upload
egy_cv_debug_db_full_no_build: conj_cv_egy_db_full egy_cv_bank_annotation egy_cv_bank_upload egy_cv_auto_qc_upload

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