
import numpy as np 
import xlrd 
import pandas as pd

"""
Reads the weekly statics for a given region
"""
def read_excel(filename, region_name):

    sheet_name = 'Veckodata Region'
    df = pd.read_excel(filename, sheet_name = sheet_name,engine='openpyxl')

    region_rows = df.loc[df.Region==region_name]

    week_nrs = region_rows.veckonummer
    confirmed_cases = region_rows.Antal_fall_vecka
    iva_cases = region_rows.Antal_intensivvårdade_vecka
    years = region_rows.år

    week_nrs = week_nrs.to_numpy()
    confirmed_cases = confirmed_cases.to_numpy()
    iva_cases = iva_cases.to_numpy()
    years = years.to_numpy()

    week_nrs += (years > 2021)*53

    indx = np.argsort(week_nrs)

    week_nrs = week_nrs[indx]
    confirmed_cases = confirmed_cases[indx]
    iva_cases =iva_cases[indx]

    return (week_nrs, confirmed_cases, iva_cases)


def read_iva_excel(filename):

    sheet_name = 'SIRI-portalen'
    book = xlrd.open_workbook(filename)
    sheet = book.sheet_by_index(0)

    nr_cases = []
    dates = []
    row = 3
    while True:
        try:
            val = sheet.cell_value(rowx=row, colx=1)
            date = sheet.cell_value(rowx=row, colx=0)
        except:
            break

        nr_cases.append(int(val))
        dates.append(date)

        row +=1

    return (dates,np.array(nr_cases))



