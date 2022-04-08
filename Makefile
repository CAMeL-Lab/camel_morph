camel_tools = "/Users/chriscay/Library/Mobile Documents/com~apple~CloudDocs/NYUAD/camel_tools"
service_account = "/Users/chriscay/.config/gspread/service_account.json"

make_db_pv_msa:
	python db_maker.py -config_file config.json -config_name pv_msa_order-v4_red -output_dir db_iterations_local -camel_tools $(camel_tools)
make_db_iv_msa:
	python db_maker.py -config_file config.json -config_name iv_msa_order-v4_red -output_dir db_iterations_local -camel_tools $(camel_tools)
make_db_cv_msa:	
	python db_maker.py -config_file config.json -config_name cv_msa_order-v4_red -output_dir db_iterations_local -camel_tools $(camel_tools)
make_db_all_msa_red:
	python db_maker.py -config_file config.json -config_name all_msa_order-v4_red -output_dir db_iterations_local -camel_tools $(camel_tools)
make_db_all_msa:
	python db_maker.py -config_file config.json -config_name all_msa_order-v4 -output_dir db_iterations_local -camel_tools $(camel_tools)

make_db_pv_glf:	
	python db_maker.py -config_file config.json -config_name pv_glf_order-v4_red -output_dir db_iterations_local -camel_tools $(camel_tools)
make_db_iv_glf:	
	python db_maker.py -config_file config.json -config_name iv_glf_order-v4_red -output_dir db_iterations_local -camel_tools $(camel_tools)
make_db_cv_glf:	
	python db_maker.py -config_file config.json -config_name cv_glf_order-v4_red -output_dir db_iterations_local -camel_tools $(camel_tools)
make_db_pv_egy:	
	python db_maker.py -config_file config.json -config_name pv_egy_order-v4_red -output_dir db_iterations_local -camel_tools $(camel_tools)
make_db_iv_egy:	
	python db_maker.py -config_file config.json -config_name iv_egy_order-v4_red -output_dir db_iterations_local -camel_tools $(camel_tools)
make_db_cv_egy:	
	python db_maker.py -config_file config.json -config_name cv_egy_order-v4_red -output_dir db_iterations_local -camel_tools $(camel_tools)
make_db_nom_msa:	
	python db_maker.py -config_file config.json -config_name nom_msa_red -output_dir db_iterations_local -camel_tools $(camel_tools)

repr_lemmas_pv_msa:
	python create_repr_lemmas_list.py -config_file config.json -config_name pv_msa_order-v4_red -output_name repr_lemmas_pv_msa.pkl -output_dir conjugation_local/repr_lemmas -pos_type verbal -display_format expanded -bank_dir conjugation_local/banks -bank_name MSA-PV-Bank -camel_tools $(camel_tools)
repr_lemmas_iv_msa:	
	python create_repr_lemmas_list.py -config_file config.json -config_name iv_msa_order-v3 -output_name repr_lemmas_iv_msa.pkl -output_dir conjugation_local/repr_lemmas -pos_type verbal -display_format expanded -bank_dir conjugation_local/banks -bank_name MSA-IV-Ind-Bank -camel_tools $(camel_tools)
repr_lemmas_cv_msa:	
	python create_repr_lemmas_list.py -config_file config.json -config_name cv_msa_order-v4_red -output_name repr_lemmas_cv_msa.pkl -output_dir conjugation_local/repr_lemmas -pos_type verbal -display_format expanded -bank_dir conjugation_local/banks -bank_name MSA-CV-Ind-Bank -camel_tools $(camel_tools)
repr_lemmas_pv_glf:	
	python create_repr_lemmas_list.py -config_file config.json -config_name pv_glf_order-v4_red -output_name repr_lemmas_pv_glf.pkl -output_dir conjugation_local/repr_lemmas -pos_type verbal -display_format expanded -bank_dir conjugation_local/banks -bank_name GLF-PV-Bank -lexprob data/glf_verbs_lexprob.tsv -camel_tools $(camel_tools)
repr_lemmas_iv_glf:	
	python create_repr_lemmas_list.py -config_file config.json -config_name iv_glf_order-v4_red -output_name repr_lemmas_iv_glf.pkl -output_dir conjugation_local/repr_lemmas -pos_type verbal -display_format expanded -bank_dir conjugation_local/banks -bank_name GLF-IV-Bank -lexprob data/glf_verbs_lexprob.tsv -camel_tools $(camel_tools)
repr_lemmas_cv_glf:	
	python create_repr_lemmas_list.py -config_file config.json -config_name cv_glf_order-v4_red -output_name repr_lemmas_cv_glf.pkl -output_dir conjugation_local/repr_lemmas -pos_type verbal -display_format expanded -bank_dir conjugation_local/banks -bank_name GLF-CV-Bank -lexprob data/glf_verbs_lexprob.tsv -camel_tools $(camel_tools)
repr_lemmas_pv_egy:	
	python create_repr_lemmas_list.py -config_file config.json -config_name pv_egy_order-v4_red -output_name repr_lemmas_pv_egy.pkl -output_dir conjugation_local/repr_lemmas -pos_type verbal -display_format expanded -bank_dir conjugation_local/banks -bank_name EGY-PV-Bank -lexprob data/egy_verbs_lexprob.tsv -camel_tools $(camel_tools)
repr_lemmas_iv_egy:	
	python create_repr_lemmas_list.py -config_file config.json -config_name iv_egy_order-v4_red -output_name repr_lemmas_iv_egy.pkl -output_dir conjugation_local/repr_lemmas -pos_type verbal -display_format expanded -bank_dir conjugation_local/banks -bank_name EGY-IV-Bank -lexprob data/egy_verbs_lexprob.tsv -camel_tools $(camel_tools)
repr_lemmas_cv_egy:	
	python create_repr_lemmas_list.py -config_file config.json -config_name cv_egy_order-v4_red -output_name repr_lemmas_cv_egy.pkl -output_dir conjugation_local/repr_lemmas -pos_type verbal -display_format expanded -bank_dir conjugation_local/banks -bank_name EGY-CV-Bank -lexprob data/egy_verbs_lexprob.tsv -camel_tools $(camel_tools)
repr_lemmas_nom_msa:	
	python create_repr_lemmas_list.py -config_file config.json -config_name nom_msa_split -output_name repr_lemmas_nom_msa.pkl -output_dir conjugation_local/repr_lemmas -pos_type nominal -display_format expanded -bank_dir conjugation_local/banks -bank_name MSA-Nom-Bank -camel_tools $(camel_tools)

conj_pv_msa:
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_pv_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_pv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp p -mod i -dialect msa -output_name conj_pv_msa_v1.0.tsv -output_dir conjugation_local/tables -camel_tools $(camel_tools)

conj_iv_i_msa:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_iv_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_iv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp i -mod i -dialect msa -output_name conj_iv-i_msa_v1.0.tsv -output_dir conjugation_local/tables -camel_tools $(camel_tools)
conj_iv_s_msa:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_iv_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_iv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp i -mod s -dialect msa -output_name conj_iv-s_msa_v1.0.tsv -output_dir conjugation_local/tables -camel_tools $(camel_tools)
conj_iv_j_msa:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_iv_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_iv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp i -mod j -dialect msa -output_name conj_iv-j_msa_v1.0.tsv -output_dir conjugation_local/tables -camel_tools $(camel_tools)
conj_iv_e_msa:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_iv_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_iv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp i -mod e -dialect msa -output_name conj_iv-e_msa_v1.0.tsv -output_dir conjugation_local/tables -camel_tools $(camel_tools)
conj_iv_x_msa:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_iv_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_iv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp i -mod x -dialect msa -output_name conj_iv-x_msa_v1.0.tsv -output_dir conjugation_local/tables -camel_tools $(camel_tools)

conj_cv_i_msa:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_cv_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_cv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp c -mod i -dialect msa -output_name conj_cv-i_msa_v1.0.tsv -output_dir conjugation_local/tables -camel_tools $(camel_tools)
conj_cv_e_msa:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_cv_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_cv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp c -mod e -dialect msa -output_name conj_cv-e_msa_v1.0.tsv -output_dir conjugation_local/tables -camel_tools $(camel_tools)
conj_cv_x_msa:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_cv_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_cv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp c -mod x -dialect msa -output_name conj_cv-x_msa_v1.0.tsv -output_dir conjugation_local/tables -camel_tools $(camel_tools)

conj_pv_msa_db_full:
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_pv_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_all_v1.0.db -db_dir db_iterations_local -pos_type verbal -asp p -mod i -dialect msa -output_name conj_pv_msa_v1.0.tsv -output_dir conjugation_local/tables -camel_tools $(camel_tools)

conj_iv_i_msa_db_full:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_iv_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_all_v1.0.db -db_dir db_iterations_local -pos_type verbal -asp i -mod i -dialect msa -output_name conj_iv-i_msa_v1.0.tsv -output_dir conjugation_local/tables -camel_tools $(camel_tools)
conj_iv_s_msa_db_full:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_iv_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_all_v1.0.db -db_dir db_iterations_local -pos_type verbal -asp i -mod s -dialect msa -output_name conj_iv-s_msa_v1.0.tsv -output_dir conjugation_local/tables -camel_tools $(camel_tools)
conj_iv_j_msa_db_full:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_iv_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_all_v1.0.db -db_dir db_iterations_local -pos_type verbal -asp i -mod j -dialect msa -output_name conj_iv-j_msa_v1.0.tsv -output_dir conjugation_local/tables -camel_tools $(camel_tools)
conj_iv_e_msa_db_full:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_iv_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_all_v1.0.db -db_dir db_iterations_local -pos_type verbal -asp i -mod e -dialect msa -output_name conj_iv-e_msa_v1.0.tsv -output_dir conjugation_local/tables -camel_tools $(camel_tools)
conj_iv_x_msa_db_full:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_iv_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_all_v1.0.db -db_dir db_iterations_local -pos_type verbal -asp i -mod x -dialect msa -output_name conj_iv-x_msa_v1.0.tsv -output_dir conjugation_local/tables -camel_tools $(camel_tools)

conj_cv_i_msa_db_full:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_cv_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_all_v1.0.db -db_dir db_iterations_local -pos_type verbal -asp c -mod i -dialect msa -output_name conj_cv-i_msa_v1.0.tsv -output_dir conjugation_local/tables -camel_tools $(camel_tools)
conj_cv_e_msa_db_full:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_cv_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_all_v1.0.db -db_dir db_iterations_local -pos_type verbal -asp c -mod e -dialect msa -output_name conj_cv-e_msa_v1.0.tsv -output_dir conjugation_local/tables -camel_tools $(camel_tools)
conj_cv_x_msa_db_full:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_cv_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_all_v1.0.db -db_dir db_iterations_local -pos_type verbal -asp c -mod x -dialect msa -output_name conj_cv-x_msa_v1.0.tsv -output_dir conjugation_local/tables -camel_tools $(camel_tools)

conj_nom_msa:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_nom_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_nom_v1.0_red.db -db_dir db_iterations_local -pos_type nominal -dialect msa -output_name conj_nom_msa_v1.0.tsv -output_dir conjugation_local/tables -camel_tools $(camel_tools)

conj_pv_glf:
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_pv_glf.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_glf_pv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp p -dialect glf -output_name conj_pv_glf_v1.0.tsv -output_dir conjugation_local/tables -camel_tools $(camel_tools)
conj_iv_glf:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_iv_glf.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_glf_iv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp i -dialect glf -output_name conj_iv_glf_v1.0.tsv -output_dir conjugation_local/tables -camel_tools $(camel_tools)
conj_cv_glf:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_cv_glf.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_glf_cv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp c -dialect glf -output_name conj_cv_glf_v1.0.tsv -output_dir conjugation_local/tables -camel_tools $(camel_tools)

conj_pv_egy:
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_pv_egy.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_egy_pv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp p -dialect egy -output_name conj_pv_egy_v1.0.tsv -output_dir conjugation_local/tables -camel_tools $(camel_tools)
conj_iv_egy:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_iv_egy.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_egy_iv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp i -dialect egy -output_name conj_iv_egy_v1.0.tsv -output_dir conjugation_local/tables -camel_tools $(camel_tools)
conj_cv_egy:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_cv_egy.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_egy_cv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp c -dialect egy -output_name conj_cv_egy_v1.0.tsv -output_dir conjugation_local/tables -camel_tools $(camel_tools)

download_specs:
	python download_sheets.py -specs header-morph-order-sheets MSA-MORPH-Verbs-v4-Red -service_account $(service_account)
download_msa_pv:	
	python download_sheets.py -config_file config.json -config_name pv_msa_order-v4_red -service_account $(service_account)
download_msa_iv:
	python download_sheets.py -config_file config.json -config_name iv_msa_order-v4_red -service_account $(service_account)
download_msa_iv_order-v3:
	python download_sheets.py -config_file config.json -config_name iv_msa_order-v3 -service_account $(service_account)
download_msa_cv:	
	python download_sheets.py -config_file config.json -config_name cv_msa_order-v4_red -service_account $(service_account)
download_msa_all_red:	
	python download_sheets.py -config_file config.json -config_name all_msa_order-v4_red -service_account $(service_account)
download_msa_all:	
	python download_sheets.py -config_file config.json -config_name all_msa_order-v4 -service_account $(service_account)

download_glf_pv:	
	python download_sheets.py -config_file config.json -config_name pv_glf_order-v4_red -service_account $(service_account)
download_glf_iv:	
	python download_sheets.py -config_file config.json -config_name iv_glf_order-v4_red -service_account $(service_account)
download_glf_cv:	
	python download_sheets.py -config_file config.json -config_name cv_glf_order-v4_red -service_account $(service_account)
download_egy_pv:	
	python download_sheets.py -config_file config.json -config_name pv_egy_order-v4_red -service_account $(service_account)
download_egy_iv:	
	python download_sheets.py -config_file config.json -config_name iv_egy_order-v4_red -service_account $(service_account)
download_egy_cv:	
	python download_sheets.py -config_file config.json -config_name cv_egy_order-v4_red -service_account $(service_account)
download_msa_nom:	
	python download_sheets.py -config_file config.json -config_name nom_msa_red -service_account $(service_account)

upload_pv_msa:
	python format_conj_gsheets.py -dir conjugation_local/tables -file_name conj_pv_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-PV -formatting conj_tables

upload_iv_i_msa:
	python format_conj_gsheets.py -dir conjugation_local/tables -file_name conj_iv-i_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-IV-Ind -formatting conj_tables
upload_iv_s_msa:
	python format_conj_gsheets.py -dir conjugation_local/tables -file_name conj_iv-s_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-IV-Sub -formatting conj_tables
upload_iv_j_msa:
	python format_conj_gsheets.py -dir conjugation_local/tables -file_name conj_iv-j_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-IV-Jus -formatting conj_tables
upload_iv_e_msa:
	python format_conj_gsheets.py -dir conjugation_local/tables -file_name conj_iv-e_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-IV-Ener -formatting conj_tables
upload_iv_x_msa:
	python format_conj_gsheets.py -dir conjugation_local/tables -file_name conj_iv-x_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-IV-XEner -formatting conj_tables

upload_cv_i_msa:
	python format_conj_gsheets.py -dir conjugation_local/tables -file_name conj_cv-i_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-CV-Ind -formatting conj_tables
upload_cv_e_msa:
	python format_conj_gsheets.py -dir conjugation_local/tables -file_name conj_cv-e_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-CV-Ener -formatting conj_tables
upload_cv_x_msa:
	python format_conj_gsheets.py -dir conjugation_local/tables -file_name conj_cv-x_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-CV-XEner -formatting conj_tables

upload_pv_glf:
	python format_conj_gsheets.py -dir conjugation_local/tables -file_name conj_pv_glf_v1.0.tsv -spreadsheet_name Paradigm-Debugging-Dialects -gsheet_name GLF-PV -formatting conj_tables
upload_iv_glf:
	python format_conj_gsheets.py -dir conjugation_local/tables -file_name conj_iv_glf_v1.0.tsv -spreadsheet_name Paradigm-Debugging-Dialects -gsheet_name GLF-IV -formatting conj_tables
upload_cv_glf:
	python format_conj_gsheets.py -dir conjugation_local/tables -file_name conj_cv_glf_v1.0.tsv -spreadsheet_name Paradigm-Debugging-Dialects -gsheet_name GLF-CV -formatting conj_tables
upload_pv_egy:
	python format_conj_gsheets.py -dir conjugation_local/tables -file_name conj_pv_egy_v1.0.tsv -spreadsheet_name Paradigm-Debugging-Dialects -gsheet_name EGY-PV -formatting conj_tables
upload_iv_egy:
	python format_conj_gsheets.py -dir conjugation_local/tables -file_name conj_iv_egy_v1.0.tsv -spreadsheet_name Paradigm-Debugging-Dialects -gsheet_name EGY-IV -formatting conj_tables
upload_cv_egy:
	python format_conj_gsheets.py -dir conjugation_local/tables -file_name conj_cv_egy_v1.0.tsv -spreadsheet_name Paradigm-Debugging-Dialects -gsheet_name EGY-CV -formatting conj_tables

msa_pv_process: download_msa_pv repr_lemmas_pv_msa make_db_pv_msa conj_pv_msa upload_pv_msa
msa_iv_i_process: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_i_msa upload_iv_i_msa
msa_iv_s_process: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_s_msa upload_iv_s_msa
msa_iv_j_process: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_j_msa upload_iv_j_msa
msa_iv_e_process: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_e_msa upload_iv_e_msa
msa_iv_x_process: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_x_msa upload_iv_x_msa
msa_cv_i_process: download_msa_cv repr_lemmas_cv_msa make_db_cv_msa conj_cv_i_msa upload_cv_i_msa
msa_cv_e_process: download_msa_cv repr_lemmas_cv_msa make_db_cv_msa conj_cv_e_msa upload_cv_e_msa
msa_cv_x_process: download_msa_cv repr_lemmas_cv_msa make_db_cv_msa conj_cv_x_msa upload_cv_x_msa

glf_pv_process: download_glf_pv repr_lemmas_pv_glf make_db_pv_glf conj_pv_glf upload_pv_glf
glf_iv_process: download_glf_iv repr_lemmas_iv_glf make_db_iv_glf conj_iv_glf upload_iv_glf
glf_cv_process: download_glf_cv repr_lemmas_cv_glf make_db_cv_glf conj_cv_glf upload_cv_glf

egy_pv_process: download_egy_pv repr_lemmas_pv_egy make_db_pv_egy conj_pv_egy upload_pv_egy
egy_iv_process: download_egy_iv repr_lemmas_iv_egy make_db_iv_egy conj_iv_egy upload_iv_egy
egy_cv_process: download_egy_cv repr_lemmas_cv_egy make_db_cv_egy conj_cv_egy upload_cv_egy

msa_pv_bank_annotation:
	python paradigm_debugging.py -output_name paradigm_debug_pv_msa_v1.0.tsv -output_dir conjugation_local/paradigm_debugging -gsheet MSA-PV -spreadsheet Paradigm-Debugging -bank_dir conjugation_local/banks -bank_name MSA-PV-Bank -new_conj conjugation_local/tables/conj_pv_msa_v1.0.tsv -camel_tools $(camel_tools)
msa_pv_bank_upload:
	python format_conj_gsheets.py -dir conjugation_local/banks -file_name MSA-PV-Bank.tsv -spreadsheet_name Paradigm-Banks -gsheet_name MSA-PV-Bank -formatting bank -mode backup
msa_pv_auto_qc_upload:
	python format_conj_gsheets.py -dir conjugation_local/paradigm_debugging -file_name paradigm_debug_pv_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-PV -formatting conj_tables -mode backup
msa_pv_debug: download_msa_pv repr_lemmas_pv_msa make_db_pv_msa conj_pv_msa msa_pv_bank_annotation msa_pv_bank_upload msa_pv_auto_qc_upload
msa_pv_debug_db_full: download_msa_all repr_lemmas_pv_msa make_db_all_msa conj_pv_msa_db_full msa_pv_bank_annotation msa_pv_bank_upload msa_pv_auto_qc_upload

msa_iv_i_bank_annotation:
	python paradigm_debugging.py -output_name paradigm_debug_iv-i_msa_v1.0.tsv -output_dir conjugation_local/paradigm_debugging -gsheet MSA-IV-Ind -spreadsheet Paradigm-Debugging -bank_dir conjugation_local/banks -bank_name MSA-IV-Ind-Bank -new_conj conjugation_local/tables/conj_iv-i_msa_v1.0.tsv -camel_tools $(camel_tools)
msa_iv_i_bank_upload:
	python format_conj_gsheets.py -dir conjugation_local/banks -file_name MSA-IV-Ind-Bank.tsv -spreadsheet_name Paradigm-Banks -gsheet_name MSA-IV-Ind-Bank -formatting bank -mode backup
msa_iv_i_auto_qc_upload:
	python format_conj_gsheets.py -dir conjugation_local/paradigm_debugging -file_name paradigm_debug_iv-i_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-IV-Ind -formatting conj_tables -mode backup
msa_iv_i_debug: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_i_msa msa_iv_i_bank_annotation msa_iv_i_bank_upload msa_iv_i_auto_qc_upload
msa_iv_i_debug_db_full: download_msa_all repr_lemmas_iv_msa make_db_all_msa conj_iv_i_msa_db_full msa_iv_i_bank_annotation msa_iv_i_bank_upload msa_iv_i_auto_qc_upload

msa_iv_s_bank_annotation:
	python paradigm_debugging.py -output_name paradigm_debug_iv-s_msa_v1.0.tsv -output_dir conjugation_local/paradigm_debugging -gsheet MSA-IV-Sub -spreadsheet Paradigm-Debugging -bank_dir conjugation_local/banks -bank_name MSA-IV-Sub-Bank -new_conj conjugation_local/tables/conj_iv-s_msa_v1.0.tsv -camel_tools $(camel_tools)
msa_iv_s_bank_upload:
	python format_conj_gsheets.py -dir conjugation_local/banks -file_name MSA-IV-Sub-Bank.tsv -spreadsheet_name Paradigm-Banks -gsheet_name MSA-IV-Sub-Bank -formatting bank -mode backup
msa_iv_s_auto_qc_upload:
	python format_conj_gsheets.py -dir conjugation_local/paradigm_debugging -file_name paradigm_debug_iv-s_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-IV-Sub -formatting conj_tables -mode backup
msa_iv_s_debug: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_s_msa msa_iv_s_bank_annotation msa_iv_s_bank_upload msa_iv_s_auto_qc_upload
msa_iv_s_debug_db_full: download_msa_all repr_lemmas_iv_msa make_db_all_msa conj_iv_s_msa_db_full msa_iv_s_bank_annotation msa_iv_s_bank_upload msa_iv_s_auto_qc_upload

msa_iv_j_bank_annotation:
	python paradigm_debugging.py -output_name paradigm_debug_iv-j_msa_v1.0.tsv -output_dir conjugation_local/paradigm_debugging -gsheet MSA-IV-Jus -spreadsheet Paradigm-Debugging -bank_dir conjugation_local/banks -bank_name MSA-IV-Jus-Bank -new_conj conjugation_local/tables/conj_iv-j_msa_v1.0.tsv -camel_tools $(camel_tools)
msa_iv_j_bank_upload:
	python format_conj_gsheets.py -dir conjugation_local/banks -file_name MSA-IV-Jus-Bank.tsv -spreadsheet_name Paradigm-Banks -gsheet_name MSA-IV-Jus-Bank -formatting bank -mode backup
msa_iv_j_auto_qc_upload:
	python format_conj_gsheets.py -dir conjugation_local/paradigm_debugging -file_name paradigm_debug_iv-j_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-IV-Jus -formatting conj_tables -mode backup
msa_iv_j_debug: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_j_msa msa_iv_j_bank_annotation msa_iv_j_bank_upload msa_iv_j_auto_qc_upload
msa_iv_j_debug_db_full: download_msa_all repr_lemmas_iv_msa make_db_all_msa conj_iv_j_msa_db_full msa_iv_j_bank_annotation msa_iv_j_bank_upload msa_iv_j_auto_qc_upload

msa_iv_e_bank_annotation:
	python paradigm_debugging.py -output_name paradigm_debug_iv-e_msa_v1.0.tsv -output_dir conjugation_local/paradigm_debugging -gsheet MSA-IV-Ener -spreadsheet Paradigm-Debugging -bank_dir conjugation_local/banks -bank_name "MSA-IV-(X)Ener-Bank" -new_conj conjugation_local/tables/conj_iv-e_msa_v1.0.tsv -camel_tools $(camel_tools)
msa_iv_e_bank_upload:
	python format_conj_gsheets.py -dir conjugation_local/banks -file_name "MSA-IV-(X)Ener-Bank.tsv" -spreadsheet_name Paradigm-Banks -gsheet_name "MSA-IV-(X)Ener-Bank" -formatting bank -mode backup
msa_iv_e_auto_qc_upload:
	python format_conj_gsheets.py -dir conjugation_local/paradigm_debugging -file_name paradigm_debug_iv-e_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-IV-Ener -formatting conj_tables -mode backup
msa_iv_e_debug: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_e_msa msa_iv_e_bank_annotation msa_iv_e_bank_upload msa_iv_e_auto_qc_upload
msa_iv_e_debug_db_full: download_msa_all repr_lemmas_iv_msa make_db_all_msa conj_iv_e_msa_db_full msa_iv_e_bank_annotation msa_iv_e_bank_upload msa_iv_e_auto_qc_upload

msa_iv_x_bank_annotation:
	python paradigm_debugging.py -output_name paradigm_debug_iv-x_msa_v1.0.tsv -output_dir conjugation_local/paradigm_debugging -gsheet MSA-IV-XEner -spreadsheet Paradigm-Debugging -bank_dir conjugation_local/banks -bank_name "MSA-IV-(X)Ener-Bank" -new_conj conjugation_local/tables/conj_iv-x_msa_v1.0.tsv -process_key extra_energetic -camel_tools $(camel_tools)
msa_iv_x_bank_upload:
	python format_conj_gsheets.py -dir conjugation_local/banks -file_name "MSA-IV-(X)Ener-Bank.tsv" -spreadsheet_name Paradigm-Banks -gsheet_name "MSA-IV-(X)Ener-Bank" -formatting bank -mode backup
msa_iv_x_auto_qc_upload:
	python format_conj_gsheets.py -dir conjugation_local/paradigm_debugging -file_name paradigm_debug_iv-x_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-IV-XEner -formatting conj_tables -mode backup
msa_iv_x_debug: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv_x_msa msa_iv_x_bank_annotation msa_iv_x_bank_upload msa_iv_x_auto_qc_upload
msa_iv_x_debug_db_full: download_msa_all repr_lemmas_iv_msa make_db_all_msa conj_iv_x_msa_db_full msa_iv_x_bank_annotation msa_iv_x_bank_upload msa_iv_x_auto_qc_upload

msa_cv_i_bank_annotation:
	python paradigm_debugging.py -output_name paradigm_debug_cv-i_msa_v1.0.tsv -output_dir conjugation_local/paradigm_debugging -gsheet MSA-CV-Ind -spreadsheet Paradigm-Debugging -bank_dir conjugation_local/banks -bank_name MSA-CV-Ind-Bank -new_conj conjugation_local/tables/conj_cv-i_msa_v1.0.tsv -camel_tools $(camel_tools)
msa_cv_i_bank_upload:
	python format_conj_gsheets.py -dir conjugation_local/banks -file_name MSA-CV-Ind-Bank.tsv -spreadsheet_name Paradigm-Banks -gsheet_name MSA-CV-Ind-Bank -formatting bank -mode backup
msa_cv_i_auto_qc_upload:
	python format_conj_gsheets.py -dir conjugation_local/paradigm_debugging -file_name paradigm_debug_cv-i_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-CV-Ind -formatting conj_tables -mode backup
msa_cv_i_debug: download_msa_cv repr_lemmas_cv_msa make_db_cv_msa conj_cv_i_msa msa_cv_i_bank_annotation msa_cv_i_bank_upload msa_cv_i_auto_qc_upload
msa_cv_i_debug_db_full: download_msa_all repr_lemmas_cv_msa make_db_all_msa conj_cv_i_msa_db_full msa_cv_i_bank_annotation msa_cv_i_bank_upload msa_cv_i_auto_qc_upload

msa_cv_e_bank_annotation:
	python paradigm_debugging.py -output_name paradigm_debug_cv-e_msa_v1.0.tsv -output_dir conjugation_local/paradigm_debugging -gsheet MSA-CV-Ener -spreadsheet Paradigm-Debugging -bank_dir conjugation_local/banks -bank_name "MSA-CV-(X)Ener-Bank" -new_conj conjugation_local/tables/conj_cv-e_msa_v1.0.tsv -camel_tools $(camel_tools)
msa_cv_e_bank_upload:
	python format_conj_gsheets.py -dir conjugation_local/banks -file_name "MSA-CV-(X)Ener-Bank.tsv" -spreadsheet_name Paradigm-Banks -gsheet_name "MSA-CV-(X)Ener-Bank" -formatting bank -mode backup
msa_cv_e_auto_qc_upload:
	python format_conj_gsheets.py -dir conjugation_local/paradigm_debugging -file_name paradigm_debug_cv-e_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-CV-Ener -formatting conj_tables -mode backup
msa_cv_e_debug: download_msa_cv repr_lemmas_cv_msa make_db_cv_msa conj_cv_e_msa msa_cv_e_bank_annotation msa_cv_e_bank_upload msa_cv_e_auto_qc_upload
msa_cv_e_debug_db_full: download_msa_all repr_lemmas_cv_msa make_db_all_msa conj_cv_e_msa_db_full msa_cv_e_bank_annotation msa_cv_e_bank_upload msa_cv_e_auto_qc_upload

msa_cv_x_bank_annotation:
	python paradigm_debugging.py -output_name paradigm_debug_cv-x_msa_v1.0.tsv -output_dir conjugation_local/paradigm_debugging -gsheet MSA-CV-XEner -spreadsheet Paradigm-Debugging -bank_dir conjugation_local/banks -bank_name "MSA-CV-(X)Ener-Bank" -new_conj conjugation_local/tables/conj_cv-x_msa_v1.0.tsv -process_key extra_energetic -camel_tools $(camel_tools)
msa_cv_x_bank_upload:
	python format_conj_gsheets.py -dir conjugation_local/banks -file_name "MSA-CV-(X)Ener-Bank.tsv" -spreadsheet_name Paradigm-Banks -gsheet_name "MSA-CV-(X)Ener-Bank" -formatting bank -mode backup
msa_cv_x_auto_qc_upload:
	python format_conj_gsheets.py -dir conjugation_local/paradigm_debugging -file_name paradigm_debug_cv-x_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-CV-XEner -formatting conj_tables -mode backup
msa_cv_x_debug: download_msa_cv repr_lemmas_cv_msa make_db_cv_msa conj_cv_x_msa msa_cv_x_bank_annotation msa_cv_x_bank_upload msa_cv_x_auto_qc_upload
msa_cv_x_debug_db_full: download_msa_all repr_lemmas_cv_msa make_db_all_msa conj_cv_x_msa_db_full msa_cv_x_bank_annotation msa_cv_x_bank_upload msa_cv_x_auto_qc_upload

egy_pv_bank_annotation:
	python paradigm_debugging.py -output_name paradigm_debug_pv_egy_v1.0.tsv -output_dir conjugation_local/paradigm_debugging -gsheet EGY-PV -spreadsheet Paradigm-Debugging-Dialects -bank_dir conjugation_local/banks -bank_name EGY-PV-Bank -new_conj conjugation_local/tables/conj_pv_egy_v1.0.tsv -camel_tools $(camel_tools)
egy_pv_bank_upload:
	python format_conj_gsheets.py -dir conjugation_local/banks -file_name EGY-PV-Bank.tsv -spreadsheet_name Paradigm-Banks -gsheet_name EGY-PV-Bank -formatting bank -mode backup
egy_pv_auto_qc_upload:
	python format_conj_gsheets.py -dir conjugation_local/paradigm_debugging -file_name paradigm_debug_pv_egy_v1.0.tsv -spreadsheet_name Paradigm-Debugging-Dialects -gsheet_name EGY-PV -formatting conj_tables -mode backup
egy_pv_debug: download_egy_pv repr_lemmas_pv_egy make_db_pv_egy conj_pv_egy egy_pv_bank_annotation egy_pv_bank_upload egy_pv_auto_qc_upload

egy_iv_bank_annotation:
	python paradigm_debugging.py -output_name paradigm_debug_iv_egy_v1.0.tsv -output_dir conjugation_local/paradigm_debugging -gsheet EGY-IV -spreadsheet Paradigm-Debugging-Dialects -bank_dir conjugation_local/banks -bank_name EGY-IV-Bank -new_conj conjugation_local/tables/conj_iv_egy_v1.0.tsv -camel_tools $(camel_tools)
egy_iv_bank_upload:
	python format_conj_gsheets.py -dir conjugation_local/banks -file_name EGY-IV-Bank.tsv -spreadsheet_name Paradigm-Banks -gsheet_name EGY-IV-Bank -formatting bank -mode backup
egy_iv_auto_qc_upload:
	python format_conj_gsheets.py -dir conjugation_local/paradigm_debugging -file_name paradigm_debug_iv_egy_v1.0.tsv -spreadsheet_name Paradigm-Debugging-Dialects -gsheet_name EGY-IV -formatting conj_tables -mode backup
egy_iv_debug: download_egy_iv repr_lemmas_iv_egy make_db_iv_egy conj_iv_egy egy_iv_bank_annotation egy_iv_bank_upload egy_iv_auto_qc_upload

eval_camel_tb_compare:
	python eval/evaluate_camel_morph.py -data_path eval/camel_tb_uniq_types.txt -preprocessing camel_tb -db_dir db_iterations_local -config_file config.json -config_name all_msa_order-v4 -camel_tools $(camel_tools) -baseline_db eval/calima-msa-s31_0.4.2.utf8.db -eval_mode compare -results_path eval/camel_tb_compare.tsv -n 100000

msa_all_debug_db_full: msa_pv_debug_db_full msa_iv_i_debug_db_full msa_iv_s_debug_db_full msa_iv_j_debug_db_full msa_iv_e_debug_db_full msa_iv_x_debug_db_full msa_cv_i_debug_db_full msa_cv_e_debug_db_full msa_cv_x_debug_db_full