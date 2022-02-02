make_dbs:
	python db_maker.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name pv_msa_order-v4_red
	python db_maker.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name iv_msa_order-v4_red
	python db_maker.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name cv_msa_order-v4_red
	python db_maker.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name pv_glf_order-v4_red
	python db_maker.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name iv_glf_order-v4_red
	python db_maker.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name cv_glf_order-v4_red
	python db_maker.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name pv_egy_order-v4_red
	python db_maker.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name iv_egy_order-v4_red
	python db_maker.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name cv_egy_order-v4_red
	python db_maker.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name nom_msa_red

repr_lemmas:
	python create_repr_lemmas_list.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name pv_msa_order-v4_red -cmplx_morph "[STEM-PV]" -output_name repr_lemmas_pv_msa.csv -pos_type verbal
	python create_repr_lemmas_list.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name iv_msa_order-v3 -cmplx_morph "[STEM-IV]" -output_name repr_lemmas_iv_msa.csv -pos_type verbal
	python create_repr_lemmas_list.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name cv_msa_order-v4_red -cmplx_morph "[STEM-CV]" -output_name repr_lemmas_cv_msa.csv -pos_type verbal
	python create_repr_lemmas_list.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name pv_glf_order-v4_red -cmplx_morph "[STEM-PV]" -output_name repr_lemmas_pv_glf.csv -pos_type verbal
	python create_repr_lemmas_list.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name iv_glf_order-v4_red -cmplx_morph "[STEM-IV]" -output_name repr_lemmas_iv_glf.csv -pos_type verbal
	python create_repr_lemmas_list.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name cv_glf_order-v4_red -cmplx_morph "[STEM-CV]" -output_name repr_lemmas_cv_glf.csv -pos_type verbal
	python create_repr_lemmas_list.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name pv_egy_order-v4_red -cmplx_morph "[STEM-PV]" -output_name repr_lemmas_pv_egy.csv -pos_type verbal
	python create_repr_lemmas_list.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name iv_egy_order-v4_red -cmplx_morph "[STEM-IV]" -output_name repr_lemmas_iv_egy.csv -pos_type verbal
	python create_repr_lemmas_list.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name cv_egy_order-v4_red -cmplx_morph "[STEM-CV]" -output_name repr_lemmas_cv_egy.csv -pos_type verbal
	python create_repr_lemmas_list.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name nom_msa_red -cmplx_morph "[STEM]" -output_name repr_lemmas_nom_msa.csv -pos_type nominal

conj_tables_msa:
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_pv_msa.csv -db XYZ_msa_pv_v1.0_red.db -pos_type verbal -asp p -dialect msa -output_name conj_pv_msa_v1.0.tsv
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_iv_msa.csv -db XYZ_msa_iv_v1.0_red.db -pos_type verbal -asp i -mod i -dialect msa -output_name conj_iv-i_msa_v1.0.tsv
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_iv_msa.csv -db XYZ_msa_iv_v1.0_red.db -pos_type verbal -asp i -mod s -dialect msa -output_name conj_iv-s_msa_v1.0.tsv
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_iv_msa.csv -db XYZ_msa_iv_v1.0_red.db -pos_type verbal -asp i -mod j -dialect msa -output_name conj_iv-j_msa_v1.0.tsv
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_cv_msa.csv -db XYZ_msa_cv_v1.0_red.db -pos_type verbal -asp c -dialect msa -output_name conj_cv_msa_v1.0.tsv
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_nom_msa.csv -db XYZ_msa_nom_v1.0_red.db -pos_type nominal -dialect msa -output_name conj_nom_msa_v1.0.tsv

conj_tables_glf:
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_pv_glf.csv -db XYZ_glf_pv_v1.0_red.db -pos_type verbal -asp p -dialect glf -output_name conj_pv_glf_v1.0.tsv
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_iv_glf.csv -db XYZ_glf_iv_v1.0_red.db -pos_type verbal -asp i -dialect glf -output_name conj_iv_glf_v1.0.tsv
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_cv_glf.csv -db XYZ_glf_cv_v1.0_red.db -pos_type verbal -asp c -dialect glf -output_name conj_cv_glf_v1.0.tsv

conj_tables_egy:
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_pv_egy.csv -db XYZ_egy_pv_v1.0_red.db -pos_type verbal -asp p -dialect egy -output_name conj_pv_egy_v1.0.tsv
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_iv_egy.csv -db XYZ_egy_iv_v1.0_red.db -pos_type verbal -asp i -dialect egy -output_name conj_iv_egy_v1.0.tsv
	python generate_conj_table.py -paradigms config_paradigms.json -repr_lemmas repr_lemmas_cv_egy.csv -db XYZ_egy_cv_v1.0_red.db -pos_type verbal -asp c -dialect egy -output_name conj_cv_egy_v1.0.tsv

conj_all: conj_tables_msa conj_tables_glf conj_tables_egy
all: make_dbs repr_lemmas conj_all

get_msa_pv:	
	python download_sheets.py -config_file config.json -config_name pv_msa_order-v4_red
get_msa_iv:
	python download_sheets.py -config_file config.json -config_name iv_msa_order-v4_red
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