import pandas as pd


# Join datasets and get needed weekly aggregates
def collect_format_data(_part_no):
    car_sales_df = pd.read_csv("car_sales.csv")
    parts_production_df = pd.read_csv("parts_production_export.csv")
    surplus_parts_df = pd.read_csv("surplus_export.csv")
    
    car_sales_df['sale_date'] = pd.to_datetime(car_sales_df['sale_date']) - pd.to_timedelta(7, unit='d')
    parts_production_df['timestamp'] = pd.to_datetime(parts_production_df['timestamp'], unit = 's') - pd.to_timedelta(7, unit='d')
    surplus_parts_df['timestamp'] = pd.to_datetime(surplus_parts_df['timestamp'], unit = 's') - pd.to_timedelta(7, unit='d')
    
    parts_production_df = parts_production_df[parts_production_df["part_no"] == _part_no] 
    surplus_parts_df = surplus_parts_df[surplus_parts_df["part_no"] == _part_no]
    
    df_final_sales = (car_sales_df
                .reset_index()
                .set_index("sale_date")
                .groupby(["model",pd.Grouper(freq='W-MON')])["VIN"].count()
                .astype(int)
                .reset_index()           
                )
    
    df_final_parts_production = (parts_production_df
                                 .reset_index()
                                 .set_index("timestamp")
                                 .groupby(["part_no",pd.Grouper(freq='W-MON')])["serial_no"].count()
                                 .astype(int)
                                 .reset_index()           
                                )
    
    df_final_surplus_parts = (surplus_parts_df
                                 .reset_index()
                                 .set_index("timestamp")
                                 .groupby(["part_no",pd.Grouper(freq='W-MON')])["serial_no"].count()
                                 .astype(int)
                                 .reset_index()           
                                )
    
    df_final_sales = df_final_sales.pivot(index = 'sale_date', columns='model',values='VIN')
    
    add_prod = df_final_sales.join(df_final_parts_production.set_index('timestamp'))
    final_df = df_final_surplus_parts.set_index('timestamp').join(add_prod,lsuffix='_surplus',rsuffix='_prod')
    final_df = final_df.rename(columns={"serial_no_surplus": "surplus_count", "serial_no_prod" : "prod_count"})
    final_df = final_df.drop(columns=['part_no_surplus','part_no_prod'])
    final_df['goal_parts'] = final_df.apply(lambda row: row.prod_count - row.surplus_count, axis = 1)
    final_df = final_df.drop(columns=['surplus_count','prod_count'])
    
    return final_df