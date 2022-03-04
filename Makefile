make_db_pv_msa:
	python db_maker.py -config_file config.json -config_name pv_msa_order-v4_red -output_dir db_iterations_local
make_db_iv_msa:
	python db_maker.py -config_file config.json -config_name iv_msa_order-v4_red -output_dir db_iterations_local
make_db_cv_msa:	
	python db_maker.py -config_file config.json -config_name cv_msa_order-v4_red -output_dir db_iterations_local
make_db_pv_glf:	
	python db_maker.py -config_file config.json -config_name pv_glf_order-v4_red -output_dir db_iterations_local
make_db_iv_glf:	
	python db_maker.py -config_file config.json -config_name iv_glf_order-v4_red -output_dir db_iterations_local
make_db_cv_glf:	
	python db_maker.py -config_file config.json -config_name cv_glf_order-v4_red -output_dir db_iterations_local
make_db_pv_egy:	
	python db_maker.py -config_file config.json -config_name pv_egy_order-v4_red -output_dir db_iterations_local
make_db_iv_egy:	
	python db_maker.py -config_file config.json -config_name iv_egy_order-v4_red -output_dir db_iterations_local
make_db_cv_egy:	
	python db_maker.py -config_file config.json -config_name cv_egy_order-v4_red -output_dir db_iterations_local
make_db_nom_msa:	
	python db_maker.py -config_file config.json -config_name nom_msa_red -output_dir db_iterations_local

make_db_verb_msa: make_db_pv_msa make_db_iv_msa make_db_cv_msa
make_db_verb_glf: make_db_pv_glf make_db_iv_glf make_db_cv_glf
make_db_verb_egy: make_db_pv_egy make_db_iv_egy make_db_cv_egy
make_db_nom_msa: make_db_nom_msa

make_db_verb: make_db_verb_msa make_db_verb_glf make_db_verb_egy
make_db_all: make_db_verb_msa make_db_verb_glf make_db_verb_egy make_db_nom_msa

repr_lemmas_pv_msa:
	python create_repr_lemmas_list.py -config_file config.json -config_name pv_msa_order-v4_red -output_name repr_lemmas_pv_msa.pkl -output_dir conjugation_local/repr_lemmas -pos_type verbal
repr_lemmas_iv_msa:	
	python create_repr_lemmas_list.py -config_file config.json -config_name iv_msa_order-v3 -output_name repr_lemmas_iv_msa.pkl -output_dir conjugation_local/repr_lemmas -pos_type verbal
repr_lemmas_cv_msa:	
	python create_repr_lemmas_list.py -config_file config.json -config_name cv_msa_order-v4_red -output_name repr_lemmas_cv_msa.pkl -output_dir conjugation_local/repr_lemmas -pos_type verbal
repr_lemmas_pv_glf:	
	python create_repr_lemmas_list.py -config_file config.json -config_name pv_glf_order-v4_red -output_name repr_lemmas_pv_glf.pkl -output_dir conjugation_local/repr_lemmas -pos_type verbal
repr_lemmas_iv_glf:	
	python create_repr_lemmas_list.py -config_file config.json -config_name iv_glf_order-v4_red -output_name repr_lemmas_iv_glf.pkl -output_dir conjugation_local/repr_lemmas -pos_type verbal
repr_lemmas_cv_glf:	
	python create_repr_lemmas_list.py -config_file config.json -config_name cv_glf_order-v4_red -output_name repr_lemmas_cv_glf.pkl -output_dir conjugation_local/repr_lemmas -pos_type verbal
repr_lemmas_pv_egy:	
	python create_repr_lemmas_list.py -config_file config.json -config_name pv_egy_order-v4_red -output_name repr_lemmas_pv_egy.pkl -output_dir conjugation_local/repr_lemmas -pos_type verbal
repr_lemmas_iv_egy:	
	python create_repr_lemmas_list.py -config_file config.json -config_name iv_egy_order-v4_red -output_name repr_lemmas_iv_egy.pkl -output_dir conjugation_local/repr_lemmas -pos_type verbal
repr_lemmas_cv_egy:	
	python create_repr_lemmas_list.py -config_file config.json -config_name cv_egy_order-v4_red -output_name repr_lemmas_cv_egy.pkl -output_dir conjugation_local/repr_lemmas -pos_type verbal
repr_lemmas_nom_msa:	
	python create_repr_lemmas_list.py -config_file config.json -config_name nom_msa_split -output_name repr_lemmas_nom_msa.pkl -output_dir conjugation_local/repr_lemmas -pos_type nominal -display_format expanded

repr_lemmas_verb_msa: repr_lemmas_pv_msa repr_lemmas_iv_msa repr_lemmas_cv_msa
repr_lemmas_verb_glf: repr_lemmas_pv_glf repr_lemmas_iv_glf repr_lemmas_cv_glf
repr_lemmas_verb_egy: repr_lemmas_pv_egy repr_lemmas_iv_egy repr_lemmas_cv_egy

repr_lemmas_verb: repr_lemmas_verb_msa repr_lemmas_verb_glf repr_lemmas_verb_egy
repr_lemmas_all: repr_lemmas_verb_msa repr_lemmas_verb_glf repr_lemmas_verb_egy repr_lemmas_nom_msa

conj_pv_msa:
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_pv_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_pv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp p -dialect msa -output_name conj_pv_msa_v1.0.tsv -output_dir conjugation_local/tables

conj_iv-i_msa:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_iv_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_iv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp i -mod i -dialect msa -output_name conj_iv-i_msa_v1.0.tsv -output_dir conjugation_local/tables
conj_iv-s_msa:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_iv_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_iv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp i -mod s -dialect msa -output_name conj_iv-s_msa_v1.0.tsv -output_dir conjugation_local/tables
conj_iv-j_msa:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_iv_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_iv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp i -mod j -dialect msa -output_name conj_iv-j_msa_v1.0.tsv -output_dir conjugation_local/tables
conj_iv-e_msa:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_iv_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_iv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp i -mod e -dialect msa -output_name conj_iv-e_msa_v1.0.tsv -output_dir conjugation_local/tables
conj_iv-x_msa:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_iv_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_iv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp i -mod x -dialect msa -output_name conj_iv-x_msa_v1.0.tsv -output_dir conjugation_local/tables

conj_cv-i_msa:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_cv_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_cv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp c -mod i -dialect msa -output_name conj_cv-i_msa_v1.0.tsv -output_dir conjugation_local/tables
conj_cv-e_msa:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_cv_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_cv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp c -mod e -dialect msa -output_name conj_cv-e_msa_v1.0.tsv -output_dir conjugation_local/tables
conj_cv-x_msa:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_cv_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_cv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp c -mod x -dialect msa -output_name conj_cv-x_msa_v1.0.tsv -output_dir conjugation_local/tables

conj_nom_msa:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_nom_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_nom_v1.0_red.db -db_dir db_iterations_local -pos_type nominal -dialect msa -output_name conj_nom_msa_v1.0.tsv -output_dir conjugation_local/tables

conj_pv_glf:
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_pv_glf.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_glf_pv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp p -dialect glf -output_name conj_pv_glf_v1.0.tsv -output_dir conjugation_local/tables
conj_iv_glf:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_iv_glf.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_glf_iv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp i -dialect glf -output_name conj_iv_glf_v1.0.tsv -output_dir conjugation_local/tables
conj_cv_glf:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_cv_glf.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_glf_cv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp c -dialect glf -output_name conj_cv_glf_v1.0.tsv -output_dir conjugation_local/tables

conj_pv_egy:
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_pv_egy.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_egy_pv_v1.0_red.db -pos_type verbal -asp p -dialect egy -output_name conj_pv_egy_v1.0.tsv -output_dir conjugation_local/tables
conj_iv_egy:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_iv_egy.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_egy_iv_v1.0_red.db -pos_type verbal -asp i -dialect egy -output_name conj_iv_egy_v1.0.tsv -output_dir conjugation_local/tables
conj_cv_egy:	
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_cv_egy.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_egy_cv_v1.0_red.db -pos_type verbal -asp c -dialect egy -output_name conj_cv_egy_v1.0.tsv -output_dir conjugation_local/tables

conj_verb_msa: conj_pv_msa conj_pv-a_msa conj_pv-p_msa conj_iv-i_msa conj_iv-s_msa conj_iv-j_msa conj_iv-e_msa conj_iv-x_msa conj_cv-i_msa conj_cv-e_msa conj_cv-x_msa
conj_verb_glf: conj_pv_glf conj_iv_glf conj_cv_glf
conj_verb_egy: conj_pv_egy conj_iv_egy conj_cv_egy

conj_verb: conj_verb_msa conj_verb_glf conj_verb_egy
conj_all: conj_verb_msa conj_verb_glf conj_verb_egy conj_nom_msa

download_specs:
	python download_sheets.py -specs header-morph-order-sheets MSA-MORPH-Verbs-v4-Red
download_msa_pv:	
	python download_sheets.py -config_file config.json -config_name pv_msa_order-v4_red
download_msa_iv:
	python download_sheets.py -config_file config.json -config_name iv_msa_order-v4_red
download_msa_iv_order-v3:
	python download_sheets.py -config_file config.json -config_name iv_msa_order-v3
download_msa_cv:	
	python download_sheets.py -config_file config.json -config_name cv_msa_order-v4_red
download_glf_pv:	
	python download_sheets.py -config_file config.json -config_name pv_glf_order-v4_red
download_glf_iv:	
	python download_sheets.py -config_file config.json -config_name iv_glf_order-v4_red
download_glf_cv:	
	python download_sheets.py -config_file config.json -config_name cv_glf_order-v4_red
download_egy_pv:	
	python download_sheets.py -config_file config.json -config_name pv_egy_order-v4_red
download_egy_iv:	
	python download_sheets.py -config_file config.json -config_name iv_egy_order-v4_red
download_egy_cv:	
	python download_sheets.py -config_file config.json -config_name cv_egy_order-v4_red
download_msa_nom:	
	python download_sheets.py -config_file config.json -config_name nom_msa_red

download_all: download_msa_pv download_msa_iv download_msa_cv download_glf_pv download_glf_iv download_glf_cv download_egy_pv download_egy_iv download_egy_cv download_msa_nom

upload_pv_msa:
	python format_conj_gsheets.py -dir conjugation_local/tables -file_name conj_pv_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-PV

upload_iv-i_msa:
	python format_conj_gsheets.py -dir conjugation_local/tables -file_name conj_iv-i_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-IV-Ind
upload_iv-s_msa:
	python format_conj_gsheets.py -dir conjugation_local/tables -file_name conj_iv-s_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-IV-Sub
upload_iv-j_msa:
	python format_conj_gsheets.py -dir conjugation_local/tables -file_name conj_iv-j_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-IV-Jus
upload_iv-e_msa:
	python format_conj_gsheets.py -dir conjugation_local/tables -file_name conj_iv-e_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-IV-Ener
upload_iv-x_msa:
	python format_conj_gsheets.py -dir conjugation_local/tables -file_name conj_iv-x_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-IV-XEner

upload_cv-i_msa:
	python format_conj_gsheets.py -dir conjugation_local/tables -file_name conj_cv-i_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-CV-Ind
upload_cv-e_msa:
	python format_conj_gsheets.py -dir conjugation_local/tables -file_name conj_cv-e_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-CV-Ener
upload_cv-x_msa:
	python format_conj_gsheets.py -dir conjugation_local/tables -file_name conj_cv-x_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-CV-XEner

upload_msa: upload_pv_msa upload_iv-i_msa upload_iv-s_msa upload_iv-j_msa upload_iv-e_msa upload_iv-x_msa upload_cv-i_msa upload_cv-e_msa upload_cv-x_msa

all: download_all make_db_all repr_lemmas_all conj_all

msa_pv_process: download_msa_pv repr_lemmas_pv_msa make_db_pv_msa conj_pv_msa upload_pv_msa
msa_iv-i_process: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv-i_msa upload_iv-i_msa
msa_iv-s_process: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv-s_msa upload_iv-s_msa
msa_iv-j_process: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv-j_msa upload_iv-j_msa
msa_iv-e_process: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv-e_msa upload_iv-e_msa
msa_iv-x_process: download_msa_iv repr_lemmas_iv_msa make_db_iv_msa conj_iv-x_msa upload_iv-x_msa
msa_cv-i_process: download_msa_cv repr_lemmas_cv_msa make_db_cv_msa conj_cv-i_msa upload_cv-i_msa
msa_cv-e_process: download_msa_cv repr_lemmas_cv_msa make_db_cv_msa conj_cv-e_msa upload_cv-e_msa
msa_cv-x_process: download_msa_cv repr_lemmas_cv_msa make_db_cv_msa conj_cv-x_msa upload_cv-x_msa

msa_pv_bank_annotation:
	python paradigm_debugging.py -output_name paradigm_debug_pv_msa_v1.0.tsv -output_dir conjugation_local/paradigm_debugging -gsheet MSA-PV -spreadsheet Paradigm-Debugging -bank_dir conjugation_local/banks -bank_name MSA-PV-Bank
msa_pv_bank_upload:
	python format_conj_gsheets.py -dir conjugation_local/banks -file_name MSA-PV-Bank.csv -spreadsheet_name Paradigm-Banks -gsheet_name MSA-PV-Bank
msa_pv_auto_qc_upload:
	python format_conj_gsheets.py -dir conjugation_local/paradigm_debugging -file_name paradigm_debug_pv_msa_v1.0.tsv -spreadsheet_name Paradigm-Debugging -gsheet_name MSA-PV

msa_pv_debug: download_msa_pv repr_lemmas_pv_msa make_db_pv_msa conj_pv_msa upload_pv_msa msa_pv_bank_annotation msa_pv_bank_upload msa_pv_auto_qc_upload



