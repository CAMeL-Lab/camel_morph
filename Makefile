make_dbs:
	python db_maker.py -config_file config.json -config_name pv_msa_order-v4_red -output_dir db_iterations_local
	python db_maker.py -config_file config.json -config_name iv_msa_order-v4_red -output_dir db_iterations_local
	python db_maker.py -config_file config.json -config_name cv_msa_order-v4_red -output_dir db_iterations_local
	python db_maker.py -config_file config.json -config_name pv_glf_order-v4_red -output_dir db_iterations_local
	python db_maker.py -config_file config.json -config_name iv_glf_order-v4_red -output_dir db_iterations_local
	python db_maker.py -config_file config.json -config_name cv_glf_order-v4_red -output_dir db_iterations_local
	python db_maker.py -config_file config.json -config_name pv_egy_order-v4_red -output_dir db_iterations_local
	python db_maker.py -config_file config.json -config_name iv_egy_order-v4_red -output_dir db_iterations_local
	python db_maker.py -config_file config.json -config_name cv_egy_order-v4_red -output_dir db_iterations_local
	python db_maker.py -config_file config.json -config_name nom_msa_red -output_dir db_iterations_local

repr_lemmas:
	python create_repr_lemmas_list.py -config_file config.json -config_name pv_msa_order-v4_red -output_name repr_lemmas_pv_msa.pkl -output_dir conjugation_local/repr_lemmas
	python create_repr_lemmas_list.py -config_file config.json -config_name iv_msa_order-v3 -output_name repr_lemmas_iv_msa.pkl -output_dir conjugation_local/repr_lemmas
	python create_repr_lemmas_list.py -config_file config.json -config_name cv_msa_order-v4_red -output_name repr_lemmas_cv_msa.pkl -output_dir conjugation_local/repr_lemmas
	python create_repr_lemmas_list.py -config_file config.json -config_name pv_glf_order-v4_red -output_name repr_lemmas_pv_glf.pkl -output_dir conjugation_local/repr_lemmas
	python create_repr_lemmas_list.py -config_file config.json -config_name iv_glf_order-v4_red -output_name repr_lemmas_iv_glf.pkl -output_dir conjugation_local/repr_lemmas
	python create_repr_lemmas_list.py -config_file config.json -config_name cv_glf_order-v4_red -output_name repr_lemmas_cv_glf.pkl -output_dir conjugation_local/repr_lemmas
	python create_repr_lemmas_list.py -config_file config.json -config_name pv_egy_order-v4_red -output_name repr_lemmas_pv_egy.pkl -output_dir conjugation_local/repr_lemmas
	python create_repr_lemmas_list.py -config_file config.json -config_name iv_egy_order-v4_red -output_name repr_lemmas_iv_egy.pkl -output_dir conjugation_local/repr_lemmas
	python create_repr_lemmas_list.py -config_file config.json -config_name cv_egy_order-v4_red -output_name repr_lemmas_cv_egy.pkl -output_dir conjugation_local/repr_lemmas
	python create_repr_lemmas_list.py -config_file config.json -config_name nom_msa_red -output_name repr_lemmas_nom_msa.pkl -output_dir conjugation_local/repr_lemmas

conj_tables_msa:
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_pv_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_pv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp p -dialect msa -output_name conj_pv_msa_v1.0.tsv -output_dir conjugation_local/tables
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_iv_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_iv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp i -mod i -dialect msa -output_name conj_iv-i_msa_v1.0.tsv -output_dir conjugation_local/tables
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_iv_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_iv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp i -mod s -dialect msa -output_name conj_iv-s_msa_v1.0.tsv -output_dir conjugation_local/tables
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_iv_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_iv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp i -mod j -dialect msa -output_name conj_iv-j_msa_v1.0.tsv -output_dir conjugation_local/tables
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_cv_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_cv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp c -dialect msa -output_name conj_cv_msa_v1.0.tsv -output_dir conjugation_local/tables
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_nom_msa.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_msa_nom_v1.0_red.db -db_dir db_iterations_local -pos_type nominal -dialect msa -output_name conj_nom_msa_v1.0.tsv -output_dir conjugation_local/tables

conj_tables_glf:
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_pv_glf.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_glf_pv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp p -dialect glf -output_name conj_pv_glf_v1.0.tsv -output_dir conjugation_local/tables
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_iv_glf.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_glf_iv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp i -dialect glf -output_name conj_iv_glf_v1.0.tsv -output_dir conjugation_local/tables
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_cv_glf.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_glf_cv_v1.0_red.db -db_dir db_iterations_local -pos_type verbal -asp c -dialect glf -output_name conj_cv_glf_v1.0.tsv -output_dir conjugation_local/tables

conj_tables_egy:
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_pv_egy.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_egy_pv_v1.0_red.db -pos_type verbal -asp p -dialect egy -output_name conj_pv_egy_v1.0.tsv -output_dir conjugation_local/tables
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_iv_egy.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_egy_iv_v1.0_red.db -pos_type verbal -asp i -dialect egy -output_name conj_iv_egy_v1.0.tsv -output_dir conjugation_local/tables
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_cv_egy.pkl -lemmas_dir conjugation_local/repr_lemmas -db XYZ_egy_cv_v1.0_red.db -pos_type verbal -asp c -dialect egy -output_name conj_cv_egy_v1.0.tsv -output_dir conjugation_local/tables

conj_all: conj_tables_msa conj_tables_glf conj_tables_egy

all: make_dbs repr_lemmas conj_all

get_msa_pv:	
	python download_sheets.py -config_file config.json -config_name pv_msa_order-v4_red
get_msa_iv:
	python download_sheets.py -config_file config.json -config_name iv_msa_order-v4_red
get_msa_iv_order-v3:
	python download_sheets.py -config_file config.json -config_name iv_msa_order-v3
get_msa_cv:	
	python download_sheets.py -config_file config.json -config_name cv_msa_order-v4_red
get_glf_pv:	
	python download_sheets.py -config_file config.json -config_name pv_glf_order-v4_red
get_glf_iv:	
	python download_sheets.py -config_file config.json -config_name iv_glf_order-v4_red
get_glf_cv:	
	python download_sheets.py -config_file config.json -config_name cv_glf_order-v4_red
get_egy_pv:	
	python download_sheets.py -config_file config.json -config_name pv_egy_order-v4_red
get_egy_iv:	
	python download_sheets.py -config_file config.json -config_name iv_egy_order-v4_red
get_egy_cv:	
	python download_sheets.py -config_file config.json -config_name cv_egy_order-v4_red
get_msa_nom:	
	python download_sheets.py -config_file config.json -config_name nom_msa_red

get_all: get_msa_pv get_msa_iv get_msa_cv get_glf_pv get_glf_iv get