import pandas as pd
def turn_dataframe_to_excel(df,filename, sheetname1="sheet1", RTL=False, df2=None, sheetname2="sheet2", df3=None, sheetname3="sheet3"):
    print("creating excel file...")
    from xlsxwriter.workbook import Workbook
    writer = pd.ExcelWriter(filename)
    df.to_excel(writer,sheetname1)
    
    #-----------right to left---------
    # Get the xlsxwriter workbook and worksheet objects.
    workbook  = writer.book
    worksheet = writer.sheets[sheetname1]

    if RTL:
        # Add the cell formats.
        format_right_to_left = workbook.add_format({'reading_order': 2})
        format_right_to_left.set_align('right')
        format_right_to_left.set_text_wrap()
        # Change the direction for the worksheet.
        worksheet.right_to_left()    
        # Make the column wider for visibility and add the reading order format.
        worksheet.set_column('C:C', 100, format_right_to_left)
        #----------------------------


    if df2 is not None: df2.to_excel(writer,sheetname2)
    if df3 is not None: df3.to_excel(writer,sheetname3)
    while True:
        try:
            writer.save()
            print(f"file {filename} is created!")
            break
        except Exception as e:
            if input("Excel file seems to be open. Close the file first. Then press anything. If 'n' enters, program stops. error:"+str(e))=='n':
                break
