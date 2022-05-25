camel_tools = "/Users/chriscay/Library/Mobile Documents/com~apple~CloudDocs/NYUAD/camel_tools"
service_account = "/Users/chriscay/.config/gspread/service_account.json"

make_db_pv_msa:
	python db_maker.py -config_name pv_msa_order-v4_red 
make_db_iv_msa:
	python db_maker.py -config_name iv_msa_order-v4_red 
make_db_cv_msa:	
	python db_maker.py -config_name cv_msa_order-v4_red 
make_db_all_msa_red:
	python db_maker.py -config_name all_msa_order-v4_red 
make_db_all_msa:
	python db_maker.py -config_name all_msa_order-v4 

make_db_pv_glf:	
	python db_maker.py -config_name pv_glf_order-v4_red 
make_db_iv_glf:	
	python db_maker.py -config_name iv_glf_order-v4_red 
make_db_cv_glf:	
	python db_maker.py -config_name cv_glf_order-v4_red 

make_db_pv_egy:	
	python db_maker.py -config_name pv_egy_order-v4_red 
make_db_iv_egy:	
	python db_maker.py -config_name iv_egy_order-v4_red 
make_db_cv_egy:	
	python db_maker.py -config_name cv_egy_order-v4_red 
make_db_all_egy:	
	python db_maker.py -config_name all_egy_order-v4 

make_db_nom_msa:	
	python db_maker.py -config_name nom_msa_red 

repr_lemmas_pv_msa:
	python create_repr_lemmas_list.py -config_name pv_msa_order-v4 -display_format expanded -feats "asp:p mod:i" -db eval/calima-msa-s31_0.4.2.utf8.db
repr_lemmas_iv_msa:	
	python create_repr_lemmas_list.py -config_name iv_msa_order-v3 -display_format expanded -feats "asp:i mod:i" -db eval/calima-msa-s31_0.4.2.utf8.db
repr_lemmas_cv_msa:	
	python create_repr_lemmas_list.py -config_name cv_msa_order-v4 -display_format expanded -feats "asp:c mod:i" -db eval/calima-msa-s31_0.4.2.utf8.db
repr_lemmas_pv_glf:	
	python create_repr_lemmas_list.py -config_name pv_glf_order-v4_red -display_format expanded -feats "asp:p mod:i"
repr_lemmas_iv_glf:	
	python create_repr_lemmas_list.py -config_name iv_glf_order-v4_red -display_format expanded -feats "asp:i mod:i"
repr_lemmas_cv_glf:	
	python create_repr_lemmas_list.py -config_name cv_glf_order-v4_red -display_format expanded -feats "asp:c mod:i"
repr_lemmas_pv_egy:	
	python create_repr_lemmas_list.py -config_name pv_egy_order-v4 -display_format expanded -feats "asp:p mod:i"
repr_lemmas_iv_egy:	
	python create_repr_lemmas_list.py -config_name iv_egy_order-v4 -display_format expanded -feats "asp:i mod:i"
repr_lemmas_cv_egy:	
	python create_repr_lemmas_list.py -config_name cv_egy_order-v4 -display_format expanded -feats "asp:c mod:i"
repr_lemmas_nom_msa:	
	python create_repr_lemmas_list.py -config_name nom_msa_split -display_format expanded -db eval/calima-msa-s31_0.4.2.utf8.db
repr_lemmas_adj_msa:
	python create_repr_lemmas_list.py -config_name adj_msa_split_red -display_format expanded -db eval/calima-msa-s31_0.4.2.utf8.db
repr_lemmas_noun_msa:
	python create_repr_lemmas_list.py -config_name noun_msa_split_red adj_msa_split_red -display_format expanded -db eval/calima-msa-s31_0.4.2.utf8.db

conj_pv_msa:
	python generate_conj_table.py -feats "asp:p mod:i" -config_name pv_msa_order-v4

conj_iv_i_msa:	
	python generate_conj_table.py -feats "asp:p mod:i" -config_name iv_msa_order-v4
conj_iv_s_msa:	
	python generate_conj_table.py -feats "asp:i mod:s" -config_name iv_msa_order-v4
conj_iv_j_msa:	
	python generate_conj_table.py -feats "asp:i mod:j" -config_name iv_msa_order-v4
conj_iv_e_msa:	
	python generate_conj_table.py -feats "asp:i mod:e" -config_name iv_msa_order-v4
conj_iv_x_msa:	
	python generate_conj_table.py -feats "asp:i mod:x" -config_name iv_msa_order-v4

conj_cv_i_msa:	
	python generate_conj_table.py -feats "asp:c mod:i" -config_name cv_msa_order-v4
conj_cv_e_msa:	
	python generate_conj_table.py -feats "asp:c mod:e" -config_name cv_msa_order-v4
conj_cv_x_msa:	
	python generate_conj_table.py -feats "asp:c mod:x" -config_name cv_msa_order-v4

conj_pv_msa_db_full:
	python generate_conj_table.py -feats "asp:p mod:i" -config_name pv_msa_order-v4 -db XYZ_msa_all_v1.0.db

conj_iv_i_msa_db_full:	
	python generate_conj_table.py -feats "asp:i mod:i" -config_name iv_msa_order-v4 -db XYZ_msa_all_v1.0.db
conj_iv_s_msa_db_full:	
	python generate_conj_table.py -feats "asp:i mod:s" -config_name iv_msa_order-v4 -db XYZ_msa_all_v1.0.db
conj_iv_j_msa_db_full:	
	python generate_conj_table.py -feats "asp:i mod:j" -config_name iv_msa_order-v4 -db XYZ_msa_all_v1.0.db
conj_iv_e_msa_db_full:	
	python generate_conj_table.py -feats "asp:i mod:e" -config_name iv_msa_order-v4 -db XYZ_msa_all_v1.0.db
conj_iv_x_msa_db_full:	
	python generate_conj_table.py -feats "asp:i mod:x" -config_name iv_msa_order-v4 -db XYZ_msa_all_v1.0.db

conj_cv_i_msa_db_full:	
	python generate_conj_table.py -feats "asp:c mod:i" -config_name cv_msa_order-v4 -db XYZ_msa_all_v1.0.db
conj_cv_e_msa_db_full:	
	python generate_conj_table.py -feats "asp:c mod:e" -config_name cv_msa_order-v4 -db XYZ_msa_all_v1.0.db
conj_cv_x_msa_db_full:	
	python generate_conj_table.py -feats "asp:c mod:x" -config_name cv_msa_order-v4 -db XYZ_msa_all_v1.0.db

conj_adj_msa:	
	python generate_conj_table.py -config_name adj_msa_split_red
conj_noun_msa:	
	python generate_conj_table.py -config_name noun_msa_split_red

conj_pv_glf:
	python generate_conj_table.py -feats "asp:p mod:i" -config_name pv_glf_order-v4_red
conj_iv_glf:	
	python generate_conj_table.py -feats "asp:i mod:i" -config_name iv_glf_order-v4_red
conj_cv_glf:	
	python generate_conj_table.py -feats "asp:c mod:i" -config_name cv_glf_order-v4_red

conj_pv_egy:
	python generate_conj_table.py -feats "asp:p mod:i" -config_name pv_egy_order-v4
conj_iv_egy:	
	python generate_conj_table.py -feats "asp:i mod:i" -config_name iv_egy_order-v4
conj_cv_egy:	
	python generate_conj_table.py -feats "asp:c mod:i" -config_name iv_egy_order-v4

conj_pv_egy_db_full:
	python generate_conj_table.py -feats "asp:p mod:i" -config_name pv_egy_order-v4 -db XYZ_egy_all_v1.0.db
conj_iv_egy_db_full:	
	python generate_conj_table.py -feats "asp:i mod:i" -config_name iv_egy_order-v4 -db XYZ_egy_all_v1.0.db
conj_cv_egy_db_full:	
	python generate_conj_table.py -feats "asp:c mod:i" -config_name cv_egy_order-v4 -db XYZ_egy_all_v1.0.db

download_specs:
	python download_sheets.py -specs header-morph-order-sheets MSA-MORPH-Verbs-v4-Red
download_msa_pv:	
	python download_sheets.py -config_name pv_msa_order-v4
download_msa_iv:
	python download_sheets.py -config_name iv_msa_order-v4
download_msa_iv_order-v3:
	python download_sheets.py -config_name iv_msa_order-v3
download_msa_cv:	
	python download_sheets.py -config_name cv_msa_order-v4
download_msa_all_red:	
	python download_sheets.py -config_name all_msa_order-v4
download_msa_all:	
	python download_sheets.py -config_name all_msa_order-v4

download_glf_pv:	
	python download_sheets.py -config_name pv_glf_order-v4_red
download_glf_iv:	
	python download_sheets.py -config_name iv_glf_order-v4_red
download_glf_cv:	
	python download_sheets.py -config_name cv_glf_order-v4_red

download_egy_pv:	
	python download_sheets.py -config_name pv_egy_order-v4
download_egy_iv:	
	python download_sheets.py -config_name iv_egy_order-v4
download_egy_cv:	
	python download_sheets.py -config_name cv_egy_order-v4
download_egy_all:	
	python download_sheets.py -config_name all_egy_order-v4

download_msa_nom:	
	python download_sheets.py -config_name nom_msa_red

upload_pv_msa:
	python format_conj_gsheets.py -formatting conj_tables -config_name pv_msa_order-v4 -feats "asp:p mod:i"

upload_iv_i_msa:
	python format_conj_gsheets.py -formatting conj_tables -config_name iv_msa_order-v4 -feats "asp:i mod:i"
upload_iv_s_msa:
	python format_conj_gsheets.py -formatting conj_tables -config_name iv_msa_order-v4 -feats "asp:i mod:s"
upload_iv_j_msa:
	python format_conj_gsheets.py -formatting conj_tables -config_name iv_msa_order-v4 -feats "asp:i mod:j"
upload_iv_e_msa:
	python format_conj_gsheets.py -formatting conj_tables -config_name iv_msa_order-v4 -feats "asp:i mod:e"
upload_iv_x_msa:
	python format_conj_gsheets.py -formatting conj_tables -config_name iv_msa_order-v4 -feats "asp:i mod:x"

upload_cv_i_msa:
	python format_conj_gsheets.py -formatting conj_tables -config_name cv_msa_order-v4 -feats "asp:c mod:i"
upload_cv_e_msa:
	python format_conj_gsheets.py -formatting conj_tables -config_name cv_msa_order-v4 -feats "asp:c mod:e"
upload_cv_x_msa:
	python format_conj_gsheets.py -formatting conj_tables -config_name cv_msa_order-v4 -feats "asp:c mod:x"

upload_adj_msa:
	python format_conj_gsheets.py -formatting conj_tables -config_name adj_msa_split_red -mode backup
upload_noun_msa:
	python format_conj_gsheets.py -formatting conj_tables -config_name noun_msa_split_red -mode backup

upload_pv_glf:
	python format_conj_gsheets.py -formatting conj_tables -config_name pv_glf_order-v4_red -feats "asp:p mod:i" -mode backup
upload_iv_glf:
	python format_conj_gsheets.py -formatting conj_tables -config_name iv_glf_order-v4_red -feats "asp:i mod:i" -mode backup
upload_cv_glf:
	python format_conj_gsheets.py -formatting conj_tables -config_name cv_glf_order-v4_red -feats "asp:c mod:i" -mode backup
upload_pv_egy:
	python format_conj_gsheets.py -formatting conj_tables -config_name pv_egy_order-v4 -feats "asp:p mod:i"
upload_iv_egy:
	python format_conj_gsheets.py -formatting conj_tables -config_name iv_egy_order-v4 -feats "asp:i mod:i"
upload_cv_egy:
	python format_conj_gsheets.py -formatting conj_tables -config_name cv_egy_order-v4 -feats "asp:c mod:i"

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
	python paradigm_debugging.py -config_name pv_msa_order-v4 -feats "asp:p mod:i"
msa_pv_bank_upload:
	python format_conj_gsheets.py -dir conjugation_local/banks -file_name MSA-PV-Bank.tsv -spreadsheet_name Paradigm-Banks -gsheet_name MSA-PV-Bank -formatting bank -mode backup
msa_pv_auto_qc_upload:
	python format_conj_gsheets.py -dir conjugation_local/paradigm_debugging -file_name paradigm_debug_pv_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-PV -formatting conj_tables -mode backup
msa_pv_debug: download_msa_pv repr_lemmas_pv_msa make_db_pv_msa conj_pv_msa msa_pv_bank_annotation msa_pv_bank_upload msa_pv_auto_qc_upload
msa_pv_debug_db_full: download_msa_all repr_lemmas_pv_msa make_db_all_msa conj_pv_msa_db_full msa_pv_bank_annotation msa_pv_bank_upload msa_pv_auto_qc_upload
msa_pv_debug_db_full_no_build: conj_pv_msa_db_full msa_pv_bank_annotation msa_pv_bank_upload msa_pv_auto_qc_upload

msa_iv_i_bank_annotation:
	python paradigm_debugging.py -config_name iv_msa_order-v4 -feats "asp:i mod:i"
msa_iv_i_bank_upload:
	python format_conj_gsheets.py -dir conjugation_local/banks -file_name MSA-IV-Ind-Bank.tsv -spreadsheet_name Paradigm-Banks -gsheet_name MSA-IV-Ind-Bank -formatting bank -mode backup
msa_iv_i_auto_qc_upload:
	python format_conj_gsheets.py -dir conjugation_local/paradigm_debugging -file_name paradigm_debug_iv-i_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-IV-Ind -formatting conj_tables -mode backup
msa_iv_i_debug: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_i_msa msa_iv_i_bank_annotation msa_iv_i_bank_upload msa_iv_i_auto_qc_upload
msa_iv_i_debug_db_full: download_msa_all repr_lemmas_iv_msa make_db_all_msa conj_iv_i_msa_db_full msa_iv_i_bank_annotation msa_iv_i_bank_upload msa_iv_i_auto_qc_upload
msa_iv_i_debug_db_full_no_build: conj_iv_i_msa_db_full msa_iv_i_bank_annotation msa_iv_i_bank_upload msa_iv_i_auto_qc_upload

msa_iv_s_bank_annotation:
	python paradigm_debugging.py -config_name iv_msa_order-v4 -feats "asp:i mod:s"
msa_iv_s_bank_upload:
	python format_conj_gsheets.py -dir conjugation_local/banks -file_name MSA-IV-Sub-Bank.tsv -spreadsheet_name Paradigm-Banks -gsheet_name MSA-IV-Sub-Bank -formatting bank -mode backup
msa_iv_s_auto_qc_upload:
	python format_conj_gsheets.py -dir conjugation_local/paradigm_debugging -file_name paradigm_debug_iv-s_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-IV-Sub -formatting conj_tables -mode backup
msa_iv_s_debug: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_s_msa msa_iv_s_bank_annotation msa_iv_s_bank_upload msa_iv_s_auto_qc_upload
msa_iv_s_debug_db_full: download_msa_all repr_lemmas_iv_msa make_db_all_msa conj_iv_s_msa_db_full msa_iv_s_bank_annotation msa_iv_s_bank_upload msa_iv_s_auto_qc_upload
msa_iv_s_debug_db_full_no_build: conj_iv_s_msa_db_full msa_iv_s_bank_annotation msa_iv_s_bank_upload msa_iv_s_auto_qc_upload

msa_iv_j_bank_annotation:
	python paradigm_debugging.py -config_name iv_msa_order-v4 -feats "asp:i mod:j"
msa_iv_j_bank_upload:
	python format_conj_gsheets.py -dir conjugation_local/banks -file_name MSA-IV-Jus-Bank.tsv -spreadsheet_name Paradigm-Banks -gsheet_name MSA-IV-Jus-Bank -formatting bank -mode backup
msa_iv_j_auto_qc_upload:
	python format_conj_gsheets.py -dir conjugation_local/paradigm_debugging -file_name paradigm_debug_iv-j_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-IV-Jus -formatting conj_tables -mode backup
msa_iv_j_debug: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_j_msa msa_iv_j_bank_annotation msa_iv_j_bank_upload msa_iv_j_auto_qc_upload
msa_iv_j_debug_db_full: download_msa_all repr_lemmas_iv_msa make_db_all_msa conj_iv_j_msa_db_full msa_iv_j_bank_annotation msa_iv_j_bank_upload msa_iv_j_auto_qc_upload
msa_iv_j_debug_db_full_no_build: conj_iv_j_msa_db_full msa_iv_j_bank_annotation msa_iv_j_bank_upload msa_iv_j_auto_qc_upload

msa_iv_e_bank_annotation:
	python paradigm_debugging.py -config_name iv_msa_order-v4 -feats "asp:i mod:e"
msa_iv_e_bank_upload:
	python format_conj_gsheets.py -dir conjugation_local/banks -file_name "MSA-IV-(X)Ener-Bank.tsv" -spreadsheet_name Paradigm-Banks -gsheet_name "MSA-IV-(X)Ener-Bank" -formatting bank -mode backup
msa_iv_e_auto_qc_upload:
	python format_conj_gsheets.py -dir conjugation_local/paradigm_debugging -file_name paradigm_debug_iv-e_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-IV-Ener -formatting conj_tables -mode backup
msa_iv_e_debug: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_e_msa msa_iv_e_bank_annotation msa_iv_e_bank_upload msa_iv_e_auto_qc_upload
msa_iv_e_debug_db_full: download_msa_all repr_lemmas_iv_msa make_db_all_msa conj_iv_e_msa_db_full msa_iv_e_bank_annotation msa_iv_e_bank_upload msa_iv_e_auto_qc_upload
msa_iv_e_debug_db_full_no_build: conj_iv_e_msa_db_full msa_iv_e_bank_annotation msa_iv_e_bank_upload msa_iv_e_auto_qc_upload

msa_iv_x_bank_annotation:
	python paradigm_debugging.py -config_name iv_msa_order-v4 -feats "asp:i mod:x" -process_key extra_energetic
msa_iv_x_bank_upload:
	python format_conj_gsheets.py -dir conjugation_local/banks -file_name "MSA-IV-(X)Ener-Bank.tsv" -spreadsheet_name Paradigm-Banks -gsheet_name "MSA-IV-(X)Ener-Bank" -formatting bank -mode backup
msa_iv_x_auto_qc_upload:
	python format_conj_gsheets.py -dir conjugation_local/paradigm_debugging -file_name paradigm_debug_iv-x_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-IV-XEner -formatting conj_tables -mode backup
msa_iv_x_debug: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_x_msa msa_iv_x_bank_annotation msa_iv_x_bank_upload msa_iv_x_auto_qc_upload
msa_iv_x_debug_db_full: download_msa_all repr_lemmas_iv_msa make_db_all_msa conj_iv_x_msa_db_full msa_iv_x_bank_annotation msa_iv_x_bank_upload msa_iv_x_auto_qc_upload
msa_iv_x_debug_db_full_no_build: conj_iv_x_msa_db_full msa_iv_x_bank_annotation msa_iv_x_bank_upload msa_iv_x_auto_qc_upload

msa_cv_i_bank_annotation:
	python paradigm_debugging.py -config_name cv_msa_order-v4 -feats "asp:c mod:i"
msa_cv_i_bank_upload:
	python format_conj_gsheets.py -dir conjugation_local/banks -file_name MSA-CV-Ind-Bank.tsv -spreadsheet_name Paradigm-Banks -gsheet_name MSA-CV-Ind-Bank -formatting bank -mode backup
msa_cv_i_auto_qc_upload:
	python format_conj_gsheets.py -dir conjugation_local/paradigm_debugging -file_name paradigm_debug_cv-i_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-CV-Ind -formatting conj_tables -mode backup
msa_cv_i_debug: download_msa_cv repr_lemmas_cv_msa make_db_cv_msa conj_cv_i_msa msa_cv_i_bank_annotation msa_cv_i_bank_upload msa_cv_i_auto_qc_upload
msa_cv_i_debug_db_full: download_msa_all repr_lemmas_cv_msa make_db_all_msa conj_cv_i_msa_db_full msa_cv_i_bank_annotation msa_cv_i_bank_upload msa_cv_i_auto_qc_upload
msa_cv_i_debug_db_full_no_build: conj_cv_i_msa_db_full msa_cv_i_bank_annotation msa_cv_i_bank_upload msa_cv_i_auto_qc_upload

msa_cv_e_bank_annotation:
	python paradigm_debugging.py -config_name cv_msa_order-v4 -feats "asp:c mod:e"
msa_cv_e_bank_upload:
	python format_conj_gsheets.py -dir conjugation_local/banks -file_name "MSA-CV-(X)Ener-Bank.tsv" -spreadsheet_name Paradigm-Banks -gsheet_name "MSA-CV-(X)Ener-Bank" -formatting bank -mode backup
msa_cv_e_auto_qc_upload:
	python format_conj_gsheets.py -dir conjugation_local/paradigm_debugging -file_name paradigm_debug_cv-e_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-CV-Ener -formatting conj_tables -mode backup
msa_cv_e_debug: download_msa_cv repr_lemmas_cv_msa make_db_cv_msa conj_cv_e_msa msa_cv_e_bank_annotation msa_cv_e_bank_upload msa_cv_e_auto_qc_upload
msa_cv_e_debug_db_full: download_msa_all repr_lemmas_cv_msa make_db_all_msa conj_cv_e_msa_db_full msa_cv_e_bank_annotation msa_cv_e_bank_upload msa_cv_e_auto_qc_upload
msa_cv_e_debug_db_full_no_build: conj_cv_e_msa_db_full msa_cv_e_bank_annotation msa_cv_e_bank_upload msa_cv_e_auto_qc_upload

msa_cv_x_bank_annotation:
	python paradigm_debugging.py -config_name cv_msa_order-v4 -feats "asp:c mod:x" -process_key extra_energetic
msa_cv_x_bank_upload:
	python format_conj_gsheets.py -dir conjugation_local/banks -file_name "MSA-CV-(X)Ener-Bank.tsv" -spreadsheet_name Paradigm-Banks -gsheet_name "MSA-CV-(X)Ener-Bank" -formatting bank -mode backup
msa_cv_x_auto_qc_upload:
	python format_conj_gsheets.py -dir conjugation_local/paradigm_debugging -file_name paradigm_debug_cv-x_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-CV-XEner -formatting conj_tables -mode backup
msa_cv_x_debug: download_msa_cv repr_lemmas_cv_msa make_db_cv_msa conj_cv_x_msa msa_cv_x_bank_annotation msa_cv_x_bank_upload msa_cv_x_auto_qc_upload
msa_cv_x_debug_db_full: download_msa_all repr_lemmas_cv_msa make_db_all_msa conj_cv_x_msa_db_full msa_cv_x_bank_annotation msa_cv_x_bank_upload msa_cv_x_auto_qc_upload
msa_cv_x_debug_db_full_no_build: conj_cv_x_msa_db_full msa_cv_x_bank_annotation msa_cv_x_bank_upload msa_cv_x_auto_qc_upload

msa_adj_bank_annotation:
	python paradigm_debugging.py -config_name adj_msa_split_red
msa_adj_bank_upload:
	python format_conj_gsheets.py -dir conjugation_local/banks -file_name MSA-Adj-Bank.tsv -spreadsheet_name Paradigm-Banks -gsheet_name MSA-Adj-Bank -formatting bank -mode backup
msa_adj_auto_qc_upload:
	python format_conj_gsheets.py -dir conjugation_local/paradigm_debugging -file_name paradigm_debug_adj_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging-Nominals -gsheet_name MSA-Adj -formatting conj_tables -mode backup
msa_adj_debug: download_msa_nom repr_lemmas_adj_msa make_db_nom_msa conj_adj_msa msa_adj_bank_annotation msa_adj_bank_upload msa_adj_auto_qc_upload
msa_adj_debug_db_full: download_msa_nom repr_lemmas_adj_msa make_db_nom_msa conj_adj_msa_db_full msa_adj_bank_annotation msa_adj_bank_upload msa_adj_auto_qc_upload
msa_adj_debug_db_full_no_build: conj_adj_msa_db_full msa_adj_bank_annotation msa_adj_bank_upload msa_adj_auto_qc_upload

egy_pv_bank_annotation:
	python paradigm_debugging.py -config_name pv_egy_order-v4 -feats "asp:p mod:i"
egy_pv_bank_upload:
	python format_conj_gsheets.py -dir conjugation_local/banks -file_name EGY-PV-Bank.tsv -spreadsheet_name Paradigm-Banks -gsheet_name EGY-PV-Bank -formatting bank -mode backup
egy_pv_auto_qc_upload:
	python format_conj_gsheets.py -dir conjugation_local/paradigm_debugging -file_name paradigm_debug_pv_egy_v1.0.tsv -spreadsheet_name Paradigm-Debugging-Dialects -gsheet_name EGY-PV -formatting conj_tables -mode backup
egy_pv_debug: download_egy_pv repr_lemmas_pv_egy make_db_pv_egy conj_pv_egy egy_pv_bank_annotation egy_pv_bank_upload egy_pv_auto_qc_upload
egy_pv_debug_db_full: download_egy_all repr_lemmas_pv_egy make_db_all_egy conj_pv_egy_db_full egy_pv_bank_annotation egy_pv_bank_upload egy_pv_auto_qc_upload
egy_pv_debug_db_full_no_build: conj_pv_egy_db_full egy_pv_bank_annotation egy_pv_bank_upload egy_pv_auto_qc_upload

egy_iv_bank_annotation:
	python paradigm_debugging.py -config_name iv_egy_order-v4 -feats "asp:i mod:i"
egy_iv_bank_upload:
	python format_conj_gsheets.py -dir conjugation_local/banks -file_name EGY-IV-Bank.tsv -spreadsheet_name Paradigm-Banks -gsheet_name EGY-IV-Bank -formatting bank -mode backup
egy_iv_auto_qc_upload:
	python format_conj_gsheets.py -dir conjugation_local/paradigm_debugging -file_name paradigm_debug_iv_egy_v1.0.tsv -spreadsheet_name Paradigm-Debugging-Dialects -gsheet_name EGY-IV -formatting conj_tables -mode backup
egy_iv_debug: download_egy_iv repr_lemmas_iv_egy make_db_iv_egy conj_iv_egy egy_iv_bank_annotation egy_iv_bank_upload egy_iv_auto_qc_upload
egy_iv_debug_db_full: download_egy_all repr_lemmas_iv_egy make_db_all_egy conj_iv_egy_db_full egy_iv_bank_annotation egy_iv_bank_upload egy_iv_auto_qc_upload
egy_iv_debug_db_full_no_build: conj_iv_egy_db_full egy_iv_bank_annotation egy_iv_bank_upload egy_iv_auto_qc_upload

egy_cv_bank_annotation:
	python paradigm_debugging.py -config_name cv_egy_order-v4 -feats "asp:c mod:i"
egy_cv_bank_upload:
	python format_conj_gsheets.py -dir conjugation_local/banks -file_name EGY-CV-Bank.tsv -spreadsheet_name Paradigm-Banks -gsheet_name EGY-CV-Bank -formatting bank -mode backup
egy_cv_auto_qc_upload:
	python format_conj_gsheets.py -dir conjugation_local/paradigm_debugging -file_name paradigm_debug_cv_egy_v1.0.tsv -spreadsheet_name Paradigm-Debugging-Dialects -gsheet_name EGY-CV -formatting conj_tables -mode backup
egy_cv_debug: download_egy_cv repr_lemmas_cv_egy make_db_cv_egy conj_cv_egy egy_cv_bank_annotation egy_cv_bank_upload egy_cv_auto_qc_upload
egy_cv_debug_db_full: download_egy_all repr_lemmas_cv_egy make_db_all_egy conj_cv_egy_db_full egy_cv_bank_annotation egy_cv_bank_upload egy_cv_auto_qc_upload
egy_cv_debug_db_full_no_build: conj_cv_egy_db_full egy_cv_bank_annotation egy_cv_bank_upload egy_cv_auto_qc_upload

eval_camel_tb_compare:
	python eval/evaluate_camel_morph.py -data_path eval/camel_tb_uniq_types.txt -preprocessing camel_tb -db_dir db_iterations_local -config_file config.json -config_name all_msa_order-v4 -camel_tools $(camel_tools) -baseline_db eval/calima-msa-s31_0.4.2.utf8.db -eval_mode compare -results_path eval/camel_tb_compare.tsv -n 100000

msa_all_debug_db_full: msa_pv_debug_db_full msa_iv_i_debug_db_full msa_iv_s_debug_db_full msa_iv_j_debug_db_full msa_iv_e_debug_db_full msa_iv_x_debug_db_full msa_cv_i_debug_db_full msa_cv_e_debug_db_full msa_cv_x_debug_db_full
msa_all_debug_db_full_build: download_msa_all repr_lemmas_pv_msa repr_lemmas_iv_msa repr_lemmas_cv_msa make_db_all_msa
msa_all_debug_db_full_no_build: msa_pv_debug_db_full_no_build msa_iv_i_debug_db_full_no_build msa_iv_s_debug_db_full_no_build msa_iv_j_debug_db_full_no_build msa_iv_e_debug_db_full_no_build msa_iv_x_debug_db_full_no_build msa_cv_i_debug_db_full_no_build msa_cv_e_debug_db_full_no_build msa_cv_x_debug_db_full_no_build

egy_all_debug_db_full: egy_pv_debug_db_full egy_iv_debug_db_full egy_cv_debug_db_full
egy_all_debug_db_full_build: download_egy_all repr_lemmas_pv_egy repr_lemmas_iv_egy repr_lemmas_cv_egy make_db_all_egy
egy_all_debug_db_full_no_build: egy_pv_debug_db_full_no_build egy_iv_debug_db_full_no_build egy_cv_debug_db_full_no_build

eval_camel_tb_backoff:
	python eval/evaluate_camel_morph.py -data_path eval/ATB123-train.102312.calima-msa-s31_0.3.0.magold -preprocessing magold -eval_mode recall_backoff -results_path eval/ATB-Recall-Backoff.tsv -n 100000 -db_dir db_iterations_local -config_file config.json -config_name all_msa_order-v4 -camel_tools $(camel_tools) -baseline_db eval/calima-egy-c044_0.2.0.utf8.db

eval_arz_recall:
	python eval/evaluate_camel_morph.py -data_path eval/ARZ-All-train.113012.magold -preprocessing magold -eval_mode recall -results_path eval/ARZ-Recall-v5.tsv -n 100000 -db_dir db_iterations_local -config_file config.json -config_name all_egy_order-v4 -camel_tools $(camel_tools) -baseline_db eval/calima-egy-c044_0.2.0.utf8.db -msa_analysis_union_config_name all_msa_order-v4 -input_source ldc_dediac_processed