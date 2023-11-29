import argparse
import os
import sys
import pandas as pd
from numpy import nan
from csv import Sniffer
import gspread
from gspread_formatting import CellFormat, format_cell_ranges, Color

file_path = os.path.abspath(__file__).split('/')
package_path = '/'.join(file_path[:len(file_path) - 1 - file_path[::-1].index('camel_morph')])
sys.path.insert(0, package_path)

from camel_morph.utils.utils import index2col_letter

COLUMNS_PROJECTS = dict(
    camel_morph=dict(
        columns=['LINE', 'PATTERN', 'ROOT', 'DEFINE', 'CLASS', 'LEMMA', 'FORM',
                 'BW', 'GLOSS', 'FEAT', 'COND-T', 'COND-S'],
        id='LINE'),
    maknuune=dict(
        columns=['ID', 'ROOT', 'ROOT_NTWS', 'LEMMA', 'FORM', 'CAPHI++',
                 'ANALYSIS', 'GLOSS', 'GLOSS_MSA', 'EXAMPLE_USAGE',
                 'NOTES', 'SOURCE', 'ANNOTATOR'],
        id='ID')
)

def _get_index2row(df):
    return {row[ID_INDEX]: tuple(row[:ID_INDEX] + row[ID_INDEX+1:])
            for row in df[COLUMNS].values.tolist()}

def _check_soundness_of_indexes(df):
    df_indexes = df[ID_COL].values.tolist()
    assert len(set(df_indexes)) == len(df_indexes)

def formatted_diff(sheets_df):
    src = _get_index2row(sheets_df['src'])
    tgt = _get_index2row(sheets_df['tgt'])

    color_tgt = CellFormat(backgroundColor=Color(0.714, 0.843, 0.659))
    color_src = CellFormat(backgroundColor=Color(0.918, 0.6, 0.6))

    tgt2src = {}
    for index_tgt, row_tgt in tgt.items():
        if index_tgt in src:
            row_src = src[index_tgt]
            # Row did not change
            if row_tgt == row_src:
                tgt2src[index_tgt] = 'no_edit'
            # An edit happened
            else:
                tgt2src[index_tgt] = [i for i, x in enumerate(row_tgt) if x != row_src[i]]
        # Row was added
        else:
            tgt2src[index_tgt] = 'new'

    COLUMNS_NON_ESSENTIAL = [col for col in sheets_df['tgt'].columns
                             if col not in COLUMNS_ESSENTIAL and col != ID_COL]

    COLUMNS_TGT = [ID_COL]
    for col in sheets_df['src'].columns:
        if col in COLUMNS_ESSENTIAL:
            COLUMNS_TGT += [col, f'{col}_OLD']
    end_col = index2col_letter(len(COLUMNS_TGT) + len(COLUMNS_NON_ESSENTIAL) - 1)

    col2index = {h: i for i, h in enumerate(COLUMNS_TGT)}
    output, formatting = [], []
    for i_sheet, row in sheets_df['tgt'].iterrows():
        index_tgt = row[ID_COL]
        diffs = tgt2src[index_tgt]
        output_ = []
        for i, col in enumerate(COLUMNS_ESSENTIAL):
            if diffs in ['no_edit', 'new']:
                output_ += [tgt[index_tgt][i], '']
                if diffs == 'new':
                    cell_a1_row = f'A{i_sheet + 2}:{end_col}{i_sheet + 2}'
                    formatting.append((cell_a1_row, color_tgt))
            else:
                if i in diffs:
                    output_ += [tgt[index_tgt][i], src[index_tgt][i]]
                    cell_a1_tgt = f'{index2col_letter(col2index[col])}{i_sheet + 2}'
                    cell_a1_src = f'{index2col_letter(col2index[col] + 1)}{i_sheet + 2}'
                    formatting += [(cell_a1_src, color_src), (cell_a1_tgt, color_tgt)]
                else:
                    output_ += [tgt[index_tgt][i], '']
        
        output_ += [row[col] for col in COLUMNS_NON_ESSENTIAL]
                
        output_.insert(0, index_tgt)
        output.append(output_)
    
    # Row was deleted
    unused_indexes = set(src) - set(tgt)
    for index_src in unused_indexes:
        row = sheets_df['src'][sheets_df['src']['ID'] == index_src]
        output_ = []
        for i, col in enumerate(COLUMNS_ESSENTIAL):
            output_ += ['', src[index_src][i]]
        output_ += [row[col].values[0] if row.get(col) is not None else ''
                    for col in COLUMNS_NON_ESSENTIAL]
        output_.insert(0, index_src)
        cell_a1_row = f'A{len(output) + 2}:{end_col}{len(output) + 2}'
        formatting.append((cell_a1_row, color_src))
        output.append(output_)

    output_df = pd.DataFrame(output)
    output_df.columns = COLUMNS_TGT + COLUMNS_NON_ESSENTIAL

    return output_df, formatting


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-select_for_compare_sh", default='',
                        type=str, help="Spreadsheet containing the data to select for compare (src).")
    parser.add_argument("-select_for_compare", default='',
                        type=str, help="Path of the file or name of the sheet to select for compare (src).")
    parser.add_argument("-compare_with_selected_sh", default='',
                        type=str, help="Spreadsheet containing the data to select for compare (tgt).")
    parser.add_argument("-compare_with_selected", default='',
                        type=str, help="Path of the file to compare with selected (tgt).")
    parser.add_argument("-output_sh", default='',
                        type=str, help="Spreadsheet to output the diff to.")
    parser.add_argument("-output", default='',
                        type=str, help="Path of the file or name of the sheet to output to.")
    parser.add_argument("-project", required=True,
                        type=str, help="Name of the project. Used to determine which columns to use.")
    parser.add_argument("-camel_tools", default='',
                        type=str, help="Path of the directory containing the camel_tools modules.")
    parser.add_argument("-service_account", default='',
                        type=str, help="Path of the JSON file containing the information about the service account used for the Google API.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    if args.camel_tools:
        camel_tools_dir = args.camel_tools
        sys.path.insert(0, camel_tools_dir)

    from camel_tools.utils.charmap import CharMapper

    COLUMNS = COLUMNS_PROJECTS[args.project]['columns']
    ID_COL = COLUMNS_PROJECTS[args.project]['id']
    COLUMNS_ESSENTIAL = [col for col in COLUMNS if col != ID_COL]
    COLUMNS_OUTPUT = COLUMNS + ['DIFF']
    ID_INDEX = COLUMNS.index(COLUMNS_PROJECTS[args.project]['id'])
    DIFF_INDEX = COLUMNS_OUTPUT.index('DIFF')

    ar2bw = CharMapper.builtin_mapper('ar2bw')
    bw2ar = CharMapper.builtin_mapper('bw2ar')

    sa = gspread.service_account(args.service_account)

    sh_sheets = [(args.select_for_compare_sh, args.select_for_compare),
                (args.compare_with_selected_sh, args.compare_with_selected)]
    
    sniffer = Sniffer()

    sheets_df, sheet_handles = {}, {}
    for index, (sh_name, sheet_name) in enumerate(sh_sheets):
        if sh_name:
            sh = sa.open(sh_name)
            sheet = sh.worksheet(sheet_name)
            sheet_handles ['src' if index == 0 else 'tgt'] = {
                'sh': sh, 'sheet': sheet}
            sheet = pd.DataFrame(sheet.get_all_records())
        else:
            with open(args.select_for_compare) as f:
                dialect = sniffer.sniff(f.readline())
            delimiter = dialect.delimiter
            sheet = pd.read_csv(sheet_name, sep=delimiter, na_filter=False)
            sheet_handles ['src' if index == 0 else 'tgt'] = {
                'sh': '', 'sheet': sheet}

        for i, row in sheet.iterrows():
            if not row[ID_COL]:
                sheet.loc[i, ID_COL] = sheet.loc[i - 1, ID_COL] + 0.001
        
        _check_soundness_of_indexes(sheet)
        sheets_df['src' if index == 0 else 'tgt'] = sheet
    
    output_df, formatting = formatted_diff(sheets_df)
    
    if not args.output_sh:
        output_df.to_csv(args.output, sep='\t')
    else:
        end_col = index2col_letter(len(output_df.columns) - 1)
        sh = sa.open(args.output_sh)
        sheets_output = sh.worksheets()
        if args.output in [sheet.title for sheet in sheets_output]:
            sheet = sh.worksheet(title=args.output)
        else:
            sheet = sh.add_worksheet(title=args.output, rows=100, cols=20)
        sheet.batch_clear([f'A:{end_col}'])
        sheet.update(
            f'A:{end_col}', [output_df.columns.values.tolist()] +
            output_df.values.tolist())
        color_reset = CellFormat(backgroundColor=Color(1.0, 1.0, 1.0))
        format_cell_ranges(sheet, [(f'A:{end_col}', color_reset)])
        formatting = [formatting[i:i+80000] for i in range(0, len(formatting), 80000)]
        for formatting_ in formatting:
            format_cell_ranges(sheet, formatting_)
        
