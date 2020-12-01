import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco
import matplotlib.pyplot as plt

#параметр форматирования данных при выводе в терминале (при желании можно дропнуть)
pd.options.display.max_rows = 500

#выкачиваем данные по всем инструмента входящим в индекс МосБиржи + качаем сам индекс МосБиржи
#Достаем из каждого дата сета цены закрытия
data = yf.download(['SBER.ME','SBERP.ME','GAZP.ME','LKOH.ME','YNDX.ME','GMKN.ME','NVTK.ME','SNGS.ME','SNGSP.ME','PLZL.ME','TATN.ME','TATNP.ME','ROSN.ME','POLY.ME','MGNT.ME','MTSS.ME','FIVE.ME','TCSG.ME','MOEX.ME','IRAO.ME','NLMK.ME','ALRS.ME','CHMF.ME','VTBR.ME','RTKM.ME','PHOR.ME','TRNFP.ME','RUAL.ME','AFKS.ME','MAGN.ME','DSKY.ME','PIKK.ME','HYDR.ME','FEES.ME','QIWI.ME','AFLT.ME','CBOM.ME','LSRG.ME','RSTI.ME','UPRO.ME'],start="2020-01-01", end="2020-11-24")
imoex = yf.download(['IMOEX.ME'],start="2020-01-01", end="2020-11-23")
closeDataImoex = imoex.Close
closeData = data.Close

# решение задачи оптимизации и поиска оптимальной границы портфелей
risk_free_rate = 0.0 # Безрисковая процентная ставка
num_periods_annually = 222 # Количество операционных дней в расчетном периоде с 2020-01-01 по 2020-11-24

returns = closeData.pct_change()
mean_returns = returns.mean()
cov_matrix = returns.cov()

#Считаем риск и доходность каждой акции в годовом исчислении
a_rsk = np.std(returns) * np.sqrt(num_periods_annually)
a_ret = mean_returns*num_periods_annually

# Функция вычисляющая риск и доходность для конкретного портфеля
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) * num_periods_annually
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(num_periods_annually)
    return std, returns

#Реализацию нахождения кф Шарпа взял с гита, так что ни на что тут не претендую))
#Функции для оптимизации по максимальному коэффициенту Шарпа
def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(neg_sharpe_ratio, num_assets * [1. / num_assets, ], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    return result


#Функции для оптимизации по минимальному риску
def portfolio_risk(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[0]

def min_risk(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(portfolio_risk, num_assets*[1./num_assets,], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)

    return result


#Функции для оптимизации по максимальной доходности
def neg_portfolio_return(weights, mean_returns, cov_matrix):
    return -1*portfolio_performance(weights, mean_returns, cov_matrix)[1]

def max_return(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(neg_portfolio_return, num_assets*[1./num_assets,], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)

    return result

#Оптимизируем. Ищем портфель с максимальным коэффициентом шарпа (sharpe_max), максимальной доходностью (return_max)
sharpe_max = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
return_max = max_return(mean_returns, cov_matrix)


#Вычисляем риски и доходности найденых портфелей
max_std, max_ret = portfolio_performance(return_max['x'], mean_returns, cov_matrix)

sdp, rp = portfolio_performance(sharpe_max['x'], mean_returns, cov_matrix)
max_sharpe_allocation = pd.DataFrame(sharpe_max.x.copy(),index=closeData.columns,columns=['allocation'])
max_sharpe_allocation.allocation = [round(i*100,2) for i in max_sharpe_allocation.allocation]
max_sharpe_allocation = max_sharpe_allocation



print("-"*80)
print("Портфель с максимальным показателем доходности\n")
print("Доходность:", round(max_ret,3))
print("Риск:", round(max_std,3))
print("Коэффициент Шарпа::", round((max_ret - risk_free_rate)/max_std, 3))
print("-"*80)
print("Стуктура портфеля\n")
print(max_sharpe_allocation)
print("-"*80)


# Построение графика всех акций и доходности конкретного портфеля
portfolio_alloc = sharpe_max['x']

index = closeData.index
p_returns = pd.DataFrame(columns=['returns'])
for day in range(1, closeData.index.size):
    # Calculating portfolio return
    date0 = index[day]
    date1 = index[day-1]
    a_return = (closeData.loc[date0] - closeData.loc[date1])/closeData.loc[date1]
    p_returns.loc[index[day]] = np.sum(a_return*portfolio_alloc)

plt.figure(figsize=(10,5))

plt.plot(p_returns.cumsum(), 'red', linewidth=3,  label='Доходность портфеля')

plt.plot(closeDataImoex.pct_change().cumsum(), 'blue', linewidth=3, label='Индекс Московской Биржи')

plt.grid(True, linestyle='--')
plt.title('Доходность портфеля (Red) и Индекс Московской Биржи(Blue)')
plt.xlabel('Дата')
plt.ylabel('Доходность')
plt.tight_layout()
plt.show()

#создаем словарь куда складываем доходности интрументов (значение) и название инструмента (ключ)
d1 = dict()
an_vol = np.std(returns) * np.sqrt(num_periods_annually)
an_rt = mean_returns * num_periods_annually
for i, txt in enumerate(closeData.columns):
    d1[txt]=round(an_rt[i],2)


#из словарю формируем датасет
#сортируем его по доходности сначала по убыванию, потом по возрастанию и выводем в обоих случаях только первые пять значений
df = pd.DataFrame.from_dict(d1, orient='index')
df = df.stack().to_frame().reset_index().drop('level_1', axis=1)
df.columns = ['Акция', 'Доходность']
print("Топ-5 инструментов по наибольшей годовой доходности\n")
print(df.sort_index().sort_values('Доходность', ascending = False).head(5).to_string(index=False))
print("-"*80)
print("Топ-5 инструментов по наименьшей годовой доходности\n")
print(df.sort_index().sort_values('Доходность', ascending = True).head(5).to_string(index=False))


#собираем новый дата сет для дневного оборота, вставляем в него названия всех инструментов из уже готового дата сета
instrument_names = df['Акция'].values.tolist()
df_new = pd.DataFrame()
print()
print("-"*80)
print("Подгрузка данных\n")
print()
#закачиваем информацию о цене открытия и объеме по каждому инструменту из нашего списка за указанный период
#по идеи можно делать подкачку данных в самом начале (просто не знаю как правильно), но надеюсь можно это делать в процессе работы программу, хотя
#при большом датасете это не очень рационально
#записываем в датасет уже готовые данные по обороту каждого инструмента за день
for name in range(40):
    a = instrument_names[name]
    data_new = yf.download(instrument_names[name], start="2020-01-01", end="2020-11-24", group_by='tickers')
    df_new[instrument_names[name]] = data_new['Open']*data_new['Volume']
print()
print("-----------Рейтинг инструменов по дневному обороту в периоде 01/01/20 - 24/11/20--------------")
print()
#пустой дата сет для записи сумарного годового оборота
ranking = pd.DataFrame()
#сумируем общую доходность по одному инструменту, фильтруем
ranking['Суммарный оборот'] = df_new.sum(axis=0)
total_trade = ranking.sort_index().sort_values('Суммарный оборот', ascending = False)
print(total_trade)



print()
print("-"*80)
print("Сводная таблица по секторам\n")

#я перевед файл их формата xlsx -> csv
#подгружаем данные их CSV файла
#интегрируем ее с предыдущем датафреймом по общей доходности
total_trade['SecurityId'] = total_trade.index
table = pd.read_csv('/Users/nikita_tililitsin/Downloads/Telegram Desktop/SecID_Sector.csv',sep=';')
C = pd.concat([total_trade,table], ignore_index=True)
total = total_trade.merge(table)
#группируем все данные по секторам
all = total.sort_index().sort_values('Sector', ascending = True)

#складываем все данные за один сектор и сортируем потом все по убыванию
df_sectors = all.groupby('Sector', as_index=False).first()
df_sectors['Суммарный оборот'] = all.groupby('Sector', as_index=False)['Суммарный оборот'].sum()['Суммарный оборот']
df_sectors.drop(['SecurityId'], axis='columns', inplace=True)
df_sectors_sort = df_sectors.sort_index().sort_values('Суммарный оборот', ascending = False)
print(df_sectors_sort.to_string(index=False))
