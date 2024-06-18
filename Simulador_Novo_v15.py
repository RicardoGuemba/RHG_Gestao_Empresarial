import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# 1. Reading and preparing data
def prepare_csv_data(file_path):
    df = pd.read_csv(file_path, sep=';', usecols=['Operação', 'Data Hora', 'Cód. Produto', 'Qtde'])
    df = df[df['Operação'] != 'Abastecimento']
    df['Data Hora'] = pd.to_datetime(df['Data Hora'], errors='coerce', dayfirst=True)

    # Remove rows where 'Data Hora' is NaT
    df = df.dropna(subset=['Data Hora'])

    df.sort_values(by='Data Hora', inplace=True)

    if df.empty:
        raise ValueError("No valid dates found after conversion. Check the format of the 'Data Hora' column.")

    first_demand_time = df['Data Hora'].min()
    last_demand_time = df['Data Hora'].max()

    if pd.isna(first_demand_time) or pd.isna(last_demand_time):
        raise ValueError("Error determining the time of the first or last demand.")

    first_period_time = first_demand_time.normalize()
    last_period_time = (last_demand_time + pd.Timedelta(days=1)).normalize()

    # Changing the interval to 1 hour
    intervals = pd.date_range(start=first_period_time, end=last_period_time, freq='1H')
    df['Interval'] = pd.cut(df['Data Hora'], bins=intervals, labels=intervals[:-1])

    df_interval = df.groupby(['Cód. Produto', 'Interval'])['Qtde'].sum().reset_index()
    df_interval['Qtde'].fillna(0, inplace=True)

    return df_interval

def read_excel_data(file_path):
    df_product = pd.read_excel(file_path, usecols=['Cód. Produto', 'VOLUME', 'Qmin', 'Qmax', 'Tipo'])
    return df_product

# 2. User input
def get_product_codes():
    product_codes = input("Enter the product codes for simulation, separated by commas: ").split(',')
    return [int(code.strip()) for code in product_codes]

# 3. Determining the best fit distribution
def best_fit_distributions(data, bins=200):
    """Model data by finding best fit distributions to data"""
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [st.norm, st.expon, st.gamma, st.beta, st.lognorm]

    results = []

    for distribution in DISTRIBUTIONS:
        try:
            with np.errstate(all='ignore'):
                params = distribution.fit(data)
                pdf = distribution.pdf(x, *params)
                sse = np.sum(np.power(y - pdf, 2.0))
                results.append((sse, distribution, params))
        except Exception:
            pass

    results.sort()
    return results  # Return all distributions

# 4. Demand simulation
def simulate_demand(df_interval, df_product, product_codes, simulation_time_hours):
    results = {}
    dist_info = {}
    total_emergency_refills = 0
    total_scheduled_refills = 0
    total_items_refilled = 0
    total_refills = 0
    refills_per_day = {code: [] for code in product_codes}
    scheduled_refills_per_hour = {hour: 0 for hour in range(24)}

    for product_code in product_codes:
        product_info = df_product[df_product['Cód. Produto'] == product_code].iloc[0]
        product_demand = df_interval[df_interval['Cód. Produto'] == product_code]

        best_fits = best_fit_distributions(product_demand['Qtde'])
        best_dist, best_dist_params = best_fits[0][1], best_fits[0][2]

        # Calculate mean and std dev from the time series
        mean_demand = product_demand['Qtde'].mean()
        std_demand = product_demand['Qtde'].std()

        stock = np.random.randint(product_info['Qmin'], product_info['Qmax'] + 1)
        stocks = [stock]
        demands = []
        qmin_times = []
        emergencies = 0
        scheduled = 0
        items_refilled = 0

        for hour in range(simulation_time_hours):
            try:
                hourly_demand = max(0, int(best_dist.rvs(*best_dist_params)))
                # Adjusting hourly demand to follow mean and std dev
                hourly_demand = int((hourly_demand - best_dist.mean(*best_dist_params)) / best_dist.std(*best_dist_params) * std_demand + mean_demand)
                hourly_demand = max(0, hourly_demand)
            except Exception as e:
                hourly_demand = mean_demand

            stock = stocks[-1] - hourly_demand
            qmin_time = (stock / hourly_demand) * 1 if hourly_demand > 0 else np.inf

            # Checking for emergency refills
            if stock <= product_info['Qmin']:
                # Refill all products with qmin_time <= 24
                for other_product_code in product_codes:
                    other_product_info = df_product[df_product['Cód. Produto'] == other_product_code].iloc[0]
                    other_stock = stocks[-1]
                    other_qmin_time = (other_stock / hourly_demand) * 1 if hourly_demand > 0 else np.inf

                    if other_qmin_time <= 24:
                        items_refilled += (other_product_info['Qmax'] - other_stock)
                        other_stock = other_product_info['Qmax']
                        emergencies += 1
                        total_emergency_refills += 1

                        # Update stock for the refilled product
                        stocks[-1] = other_stock

                items_refilled += (product_info['Qmax'] - stock)
                stock = product_info['Qmax']
                emergencies += 1
                total_emergency_refills += 1

            # Checking for scheduled refills with Qmin time condition
            product_type = product_info['Tipo']
            if (hour % 24 == 9 and product_type in ['MED', 'MAT']) or \
               (hour % 24 == 15 and product_type == 'MEC') or \
               (hour % 24 == 19 and product_type == 'GA'):
                if qmin_time <= 48:
                    items_refilled += (product_info['Qmax'] - stock)
                    stock = product_info['Qmax']
                    scheduled_refills_per_hour[hour % 24] += 1
                    if scheduled == 0:
                        scheduled += 1
                        total_scheduled_refills += 1

            stocks.append(stock)
            demands.append(hourly_demand)
            qmin_times.append(qmin_time)

            # Count refills per day
            day = hour // 24
            if day not in refills_per_day[product_code]:
                refills_per_day[product_code].append(day)

        results[product_code] = {
            'stocks': stocks[1:],  # Remove initial stock
            'demands': demands,
            'qmin_times': qmin_times,
            'emergencies': emergencies,
            'scheduled': scheduled,
            'items_refilled': items_refilled
        }

        dist_info[product_code] = {
            'distribution': best_dist.name,
            'parameters': best_dist_params
        }

        total_items_refilled += items_refilled

    total_refills = total_emergency_refills + total_scheduled_refills
    mean_daily_refills = total_refills / (simulation_time_hours / 24)
    mean_daily_items_refilled = total_items_refilled / (simulation_time_hours / 24)
    mean_scheduled_refills_per_hour = {hour: count / (simulation_time_hours / 24) for hour, count in scheduled_refills_per_hour.items()}

    return results, dist_info, {
        'total_emergency_refills': total_emergency_refills,
        'total_scheduled_refills': total_scheduled_refills,
        'total_refills': total_refills,
        'mean_daily_refills': mean_daily_refills,
        'total_items_refilled': total_items_refilled,
        'mean_daily_items_refilled': mean_daily_items_refilled,
        'refills_per_day': refills_per_day,
        'scheduled_refills_per_hour': scheduled_refills_per_hour,
        'mean_scheduled_refills_per_hour': mean_scheduled_refills_per_hour
    }

# 5. Plotting and displaying results
def plot_results(results, dist_info, df_product, simulation_time_hours, stats):
    hours = range(simulation_time_hours)

    # Plot of stocks by product type
    types = df_product['Tipo'].unique()
    for type_ in types:
        plt.figure(figsize=(12, 6))
        for product_code, result in results.items():
            product_type = df_product[df_product['Cód. Produto'] == product_code]['Tipo'].values[0]
            if product_type == type_ or (type_ in ['MED', 'MAT'] and product_type in ['MED', 'MAT']):
                plt.plot(hours, result['stocks'], label=f'Stock Product {product_code}')
        plt.title(f'Stock of {type_} Products Over Time')
        plt.xlabel('Hour')
        plt.ylabel('Stock')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Display demand values, qmin times, and items refilled
    for product_code, result in results.items():
        print(f"Product {product_code} - Distribution: {dist_info[product_code]['distribution']} with parameters {dist_info[product_code]['parameters']}")
        print(f"Demand over time for Product {product_code}: {result['demands']}")
        print(f"Qmin time over time for Product {product_code}: {result['qmin_times']}")
        print(f"Items refilled for Product {product_code}: {result['items_refilled']}")
        print(f"Refills per day for Product {product_code}: {stats['refills_per_day'][product_code]}")

    print(f"Total emergency refills: {stats['total_emergency_refills']}")
    print(f"Total scheduled refills: {stats['total_scheduled_refills']}")
    print(f"Total refills: {stats['total_refills']}")
    print(f"Mean daily refills: {stats['mean_daily_refills']}")
    print(f"Total items refilled: {stats['total_items_refilled']}")
    print(f"Mean daily items refilled: {stats['mean_daily_items_refilled']}")

    # Display scheduled refills per hour and mean scheduled refills per hour
    for hour, count in stats['scheduled_refills_per_hour'].items():
        print(f"Scheduled refills at hour {hour}: {count}")
        print(f"Mean daily scheduled refills at hour {hour}: {stats['mean_scheduled_refills_per_hour'][hour]}")

    # Plot time series and generated values for the first product
    first_product = list(results.keys())[0]
    result = results[first_product]

    plt.figure(figsize=(12, 6))
    plt.plot(hours, result['demands'], label='Real Demand')

    best_dist_name = dist_info[first_product]['distribution']
    best_dist_params = dist_info[first_product]['parameters']
    generated_values = getattr(st, best_dist_name).rvs(*best_dist_params, size=simulation_time_hours)
    generated_values = (generated_values - getattr(st, best_dist_name).mean(*best_dist_params)) / getattr(st, best_dist_name).std(*best_dist_params) * np.std(result['demands']) + np.mean(result['demands'])
    generated_values = np.maximum(0, generated_values)  # Ensure no negative values

    plt.plot(hours, generated_values, linestyle='--', label=f'{best_dist_name.capitalize()} Generated')

    plt.title(f'Time Series and Generated Values for Product {first_product}')
    plt.xlabel('Hour')
    plt.ylabel('Quantity')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Product {first_product} - Chosen distribution: {best_dist_name.capitalize()}")
    print(f"Values generated by the {best_dist_name.capitalize()} distribution:")
    print(generated_values)

# Main function to run all steps
def main():
    file_path_csv = "/Users/ricardohausguembarovski/Desktop/re_teste.csv"
    file_path_excel = "/Users/ricardohausguembarovski/Desktop/Produto_Simula_v5_c.xlsx"

    df_interval = prepare_csv_data(file_path_csv)
    df_product = read_excel_data(file_path_excel)
    product_codes = get_product_codes()

    simulation_time_hours = int(input("Enter the simulation time in hours: "))

    results, dist_info, stats = simulate_demand(df_interval, df_product, product_codes, simulation_time_hours)
    plot_results(results, dist_info, df_product, simulation_time_hours, stats)

if __name__ == "__main__":
    main()
