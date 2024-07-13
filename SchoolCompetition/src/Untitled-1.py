# %%
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from os.path import join

import warnings 
warnings.filterwarnings('ignore')

# %%
# 设置正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
# 正常显示负号
plt.rcParams['axes.unicode_minus'] = False

# %% [markdown]
# 结合我国当前的人口数量、人口结构以及人口政策，建立数学模型预测我国中小学学生数量的变化情况；

# %%
data_root = r'/home/drew/Desktop/MathModeling/SchoolCompetition/data'
cur_data = '全国数据'

# %%
# 人口出生率，死亡率和自然增长率
br_dr_ngr_df = pd.read_csv(join(data_root, cur_data, '人口出生率，死亡率和自然增长率.csv'), 
                           header=2, index_col=0).iloc[:-2, :].T
# 人口年龄结果和抚养比
pao_dr_df = pd.read_csv(join(data_root, cur_data, '人口年龄结果和抚养比.csv'), 
                        header=2, index_col=0).iloc[:-3, :].T
# 总人口
tp_df = pd.read_csv(join(data_root, cur_data, '总人口.csv'), 
                    header=2, index_col=0).iloc[:-2, :].T
# 按年龄分人口数
pbya_df = pd.read_csv(join(data_root, cur_data, '全国按年龄分人口数.csv'), 
                      header=2, index_col=0).iloc[:-2, :].T

# %% [markdown]
# 人口结构

# %%
pbya_df.columns

# %%
print(pbya_df.index)

# %%
pbya_df.head()

# %%
pbya_df.isna().any(axis=1)

# %%
modify_col = ['人口数'] + [col.split('岁')[0] for col in pbya_df.columns][1:]
pbya_df.columns = modify_col

# %%
pbya_df.index = [idx.split('年')[0] for idx in pbya_df.index]

# %%
pbya_df

# %%
tp_df.head()

# %%
plt.figure(figsize=(10, 6), dpi=300)
total_population = tp_df.iloc[:, 0]
plt.plot(
    total_population.index[::-1], 
    total_population[::-1], 
    marker='o', label='总人口数', color='teal'
)
plt.xticks(rotation=30)
plt.title('总人口数随年份变化')
plt.xlabel('年份')
plt.ylabel('人口数')
plt.legend()

# %%
drop_columns = [
    # '2005', '2015', '2023', 
    '2020', '2010'
]

# %%
plt.figure(figsize=(12, 6), dpi=300)
other_age_groups = pbya_df.iloc[1:, 1:].drop(drop_columns).apply(lambda x: x/pbya_df.iloc[:, 0])
for index, row in other_age_groups.iterrows():
    if index in drop_columns:
        continue
    x = [index.split('岁')[0] for index in row.index]
    y = row.values.reshape(-1, 1)
    plt.plot(x, y, marker='o', label=index)

plt.legend(loc='best')

plt.title('20年内不同年龄段人口分布情况')
plt.xlabel('年龄段')
plt.ylabel('人口占比分布情况')

plt.tight_layout()

plt.show()

# %%
plt.figure(figsize=(12, 6), dpi=300)
for index, row in other_age_groups.drop(drop_columns + ['2020', '2010'], axis=0).T.iterrows():
    x = [index.split('岁')[0] for index in row.index][::-1]
    y = row.values.reshape(-1, 1)[::-1]
    plt.plot(x, y, marker='o', label=index)

plt.legend(loc='lower right', bbox_to_anchor=(0.96, -0.2), ncol=10)

plt.title('不同年龄段人口20年内变化情况')
plt.xlabel('年份')
plt.ylabel('标准化后人口数分布情况')

plt.tight_layout()

plt.show()

# %%
plt.figure(figsize=(12, 6), dpi=300)
for year, color in zip(['2022', '2018', '2014', '2011'], 
                       ['slategray', 'yellow', 'coral', 'teal']):
    age_data = pbya_df.loc[year, :].drop(['人口数'])
    plt.bar(x=age_data.index,height=age_data, color=color, alpha=0.9)

plt.legend(['2022', '2018', '2014', '2011'])
plt.title('采样年各年龄段人口分布情况')
plt.xlabel('年龄段')
plt.ylabel('人口数')

plt.tight_layout()

plt.show()

# %%
br_dr_ngr_df.head()

# %%
plt.figure(figsize=(12, 6), dpi=300)
plt.axhline(y=0, color='red', linewidth=1)
for index, row in br_dr_ngr_df.T.iterrows():
    if index in drop_columns:
        continue
    x = row.index[::-1]
    y = row[::-1]
    plt.plot(x, y, marker='o', label=index)
plt.legend()
plt.show()

# %%
pbya_df.head()

# %% [markdown]
# 首先去除却缺失值,将不同年龄的分布变为占比情况, 用以消除不同尺度数据的影响

# %%
from sklearn.preprocessing import StandardScaler

# %%
train_df = pbya_df.dropna(axis=0)
std = StandardScaler()
train_df.iloc[:, 1:] = std.fit_transform(train_df.iloc[:, 1:].apply(lambda x: x/train_df.iloc[:, 0]))
train_df.drop(['人口数'], axis=1, inplace=True)
train_df.head()

# %%
from statsmodels.tsa.stattools import adfuller

# %% [markdown]
# - 正如前面所述，差分的目的是使时间序列平稳。但我们应该注意不要过度差分这个序列。
# - 过度差分的序列可能仍然是平稳的，这反过来会影响模型参数。
# - 所以我们需要确定正确的差分阶数。正确的差分阶数是使序列接近平稳所需的最小差分，该序列围绕一个确定的均值波动，并且自相关图（ACF图）很快达到零。
#   
# - 如果自相关在许多滞后期（10个或更多）为正，则序列需要进一步差分。另一方面，如果第一滞后自相关本身过于负，则序列可能过度差分了。
# - 如果我们在两个差分阶数之间无法做出决定，那么我们就选择使差分序列的标准差最小的阶数。
#   
# 现在，我们将通过以下示例来解释这些概念：
# - 首先，我将使用来自statsmodels包的增广迪基-富勒检验（ADF检验）来检查序列是否平稳。原因是只有在序列非平稳时才需要差分。否则，不需要差分，即d=0。
# - ADF检验的零假设（Ho）是时间序列是非平稳的。因此，如果检验的p值小于显著性水平（0.05），我们就拒绝零假设，并推断时间序列确实是平稳的。
# 
# 所以，在我们的情况下，如果P值 > 0.05，我们就继续寻找差分的阶数。
# 

# %%
ADF_P_VALs = {}
for col in train_df.columns:
    result = adfuller(train_df.loc[:, col].values)
    ADF_P_VALs[col] = result
adf_p_vals = pd.DataFrame(ADF_P_VALs, [
    'ADF Statistic', 'p-value', 'Lags Used', 
    'Number of Observations Used', 
    'Critical Values (1%)(5%)(10%)', 
    'icbest'
])
adf_p_vals

# %% [markdown]
# ### 表格参数解释
# 
# 这个数据表显示了多个年龄段的人口统计数据通过增广迪基-富勒（ADF）检验的结果。ADF检验是用来判断一个时间序列是否具有单位根，也就是说，它是否是非平稳的。如果时间序列是非平稳的，那么它可能需要差分以使其变得平稳，这在时间序列分析中是一个重要的步骤，特别是在应用如ARIMA模型之前。
# 
# 以下是对数据表的分析：
# 
# 1. **ADF Statistic（ADF 统计量）**：这一列显示了ADF检验的统计量。一般来说，ADF统计量越小，拒绝非平稳性的证据越强。对于大多数情况，如果ADF统计量小于临界值，我们可以认为序列是平稳的。
# 
# 2. **p-value（p 值）**：p 值是用来判断统计检验结果是否显著的。如果p值小于显著性水平（通常是0.05），我们可以拒绝零假设，认为序列是平稳的。在你提供的数据中，有些年龄段的p值远大于0.05，这意味着我们没有足够的证据拒绝序列是非平稳的零假设。
# 
# 3. **Lags Used（使用的滞后期）**：这一列显示了在进行ADF检验时使用的滞后阶数。滞后阶数的选择可以基于模型选择准则，如赤池信息准则（AIC）或贝叶斯信息准则（BIC）。
# 
# 4. **Number of Observations Used（使用的观测值数量）**：这一列显示了在ADF检验中实际使用的观测值数量。由于差分操作会减少观测值的数量，所以这个数字通常小于原始序列的长度。
# 
# 5. **Critical Values (1%)(5%)(10%)（临界值）**：这些是ADF检验的临界值，对应于不同的显著性水平。如果ADF统计量小于临界值，我们可以认为序列是平稳的。
# 
# 6. **icbest**：这是选择最佳滞后阶数时使用的信息准则值。较小的icbest值通常表示更好的模型拟合。
# 
# 根据提供的数据，我们可以得出以下结论：
# 
# - 对于年龄段 "25-29"，ADF统计量为 -4.024254，p 值为 0.001289，远小于0.05，因此我们可以认为该年龄段的序列是平稳的。
# - 年龄段 "0-4"、"5-9"、"10-14"、"15-19"、"30-34"、"35-39"、"40-44"、"45-49"、"50-54"、"55-59"、"60-64"、"65-69"、"70-74"、"75-79"、"80-84"、"85-89"、"90-94" 和 "95" 的p值都大于0.05，表明我们不能拒绝序列是非平稳的零假设。
# - 对于这些非平稳的序列，我们可能需要进一步差分。例如，"0-4" 年龄段的序列ADF统计量为 -2.672894，p 值为 0.078847，我们可能需要考虑一阶或二阶差分。
# 
# 最后，选择正确的差分阶数通常需要结合实际数据的自相关图（ACF）和偏自相关图（PACF）来决定，以及可能的模型诊断检查。
# 

# %%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# %%
fig, axes = plt.subplots(3, 2, sharex=True, figsize=(10, 8), dpi=300)
for col in train_df.columns[::-1]:
    df = train_df.loc[:, col]
    axes[0, 0].plot(df.values); axes[0, 0].set_title('Original Series')
    plot_acf(df.values, ax=axes[0, 1])

    # 1st Differencing
    axes[1, 0].plot(df.diff().values); axes[1, 0].set_title('1st Order Differencing')
    plot_acf(df.diff().dropna().values, ax=axes[1, 1])

    # 2nd Differencing
    axes[2, 0].plot(df.diff().diff().values), axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(df.diff().diff().dropna().values, ax=axes[2, 1])
plt.show()

# %% [markdown]
# 根据你提供的图像内容，这似乎是三个时间序列及其自相关函数（ACF）图的文本表示。每个部分分别展示了原始序列、一阶差分后的序列和二阶差分后的序列的自相关性。下面是对这些图的分析：
# 
# 1. **原始序列（Original Series）**:
#    - 从自相关图可以看出，原始序列的自相关值在第一个滞后（即零延迟）是1.0，这符合所有时间序列自相关图的预期，因为任何序列与自身在零延迟时完全相关。
#    - 随着滞后的增加，自相关值迅速下降，但仍然保持正值，这表明原始序列可能具有持久性（即自相关性衰减缓慢）。
# 
# 2. **一阶差分后的序列（1st Order Differencing）**:
#    - 一阶差分后的自相关图显示，在第一个滞后（即一阶差分后的零延迟）自相关值接近0.5，这表明差分后的序列仍然存在一定的自相关性。
#    - 随着滞后的增加，自相关值进一步下降，但仍然保持正值。这可能表明序列在一阶差分后仍然需要进一步的处理。
# 
# 3. **二阶差分后的序列（2nd Order Differencing）**:
#    - 二阶差分后的自相关图显示，在第一个滞后自相关值接近0.5，与一阶差分后的情况相似。
#    - 然而，值得注意的是，自相关图的模式似乎与一阶差分后的模式相似，这可能表明二阶差分并没有显著改变序列的自相关结构。
# 
# 4. **时间序列图**:
#    - 时间序列图显示了序列的波动情况。从图中可以看到，序列在一段时间内呈现出周期性的波动，这可能是季节性因素或其他周期性影响的结果。
#    - 序列的值在一定的范围内上下波动，但没有显示出明显的趋势或季节性模式。
# 
# **结论**:
# - 原始序列的自相关性衰减较慢，表明它可能是非平稳的。
# - 一阶和二阶差分后的序列自相关性有所降低，但仍然存在。这可能表明序列可能需要更多的差分或考虑其他形式的转换来实现平稳性。
# - 根据自相关图，我们不能直接得出序列是否已经平稳，可能需要进一步的分析，如偏自相关图（PACF）或再次进行ADF检验。
# 
# 请注意，这些结论是基于文本描述的简化分析，实际分析时需要查看图形并结合其他统计检验。如果你能提供实际的图形文件，我可以提供更具体的分析。
# 

# %% [markdown]
# 下一步是确定模型是否需要任何自回归（AR）项。我们将通过检查偏自相关（PACF）图来找出所需的AR项的数量。
# 偏自相关可以想象为系列与其滞后之间的相关性，在排除了中间滞后的贡献之后。因此，PACF在某种程度上传达了滞后和系列之间的纯相关性。这样，我们就可以知道是否需要将该滞后包含在AR项中。
# 系列的滞后（k）的偏自相关是Y的自回归方程中该滞后的系数。
# Yt = α0 + α1Yt−1 + α2Yt−2 + α3Yt−3
# 
# 也就是说，假设Y_t是当前系列，Y_t-1是Y的滞后1，则滞后3（Y_t-3）的偏自相关是上述方程中Y_t-3的系数α3。
# 
# 现在，我们应该找出AR项的数量。在平稳系列中的任何自相关都可以通过添加足够的AR项来纠正。因此，我们最初将AR项的阶数设置为PACF图中超过显著性极限的滞后数量。
# 
# 这段描述解释了在时间序列分析中如何使用偏自相关图来确定自回归模型的阶数。通过观察PACF图中显著不为零的滞后值，可以帮助我们识别需要包含在自回归模型中的滞后项。如果滞后值在某个点之后迅速下降至零，则该点之前的所有滞后都可能是模型中需要的AR项。
# 

# %%
fig, axes = plt.subplots(3, 2, sharex=True, figsize=(10, 8), dpi=300)
for col in train_df.columns[::-1]:
    df = train_df.loc[:, col]
    axes[0, 0].plot(df.values); axes[0, 0].set_title('Original Series')
    plot_pacf(df.values, ax=axes[0, 1])

    # 1st Differencing
    axes[1, 0].plot(df.diff().values); axes[1, 0].set_title('1st Order Differencing')
    plot_pacf(df.diff().dropna().values, ax=axes[1, 1])

    # 2nd Differencing
    axes[2, 0].plot(df.diff().diff().values), axes[2, 0].set_title('2nd Order Differencing')
    plot_pacf(df.diff().diff().dropna().values, ax=axes[2, 1])
plt.show()

# %% [markdown]
# 根据你提供的文件内容，这是三个时间序列的偏自相关函数（PACF）图的文本表示。偏自相关函数图用于识别时间序列模型的阶数，特别是自回归（AR）部分的阶数。下面是对这些图的分析：
# 
# 1. **原始序列（Original Series）**:
#    - 偏自相关图中，零延迟（即当lag=0时）的值是1.0，这是预期的，因为它代表了序列与其自身的相关性。
#    - 随着滞后增加，偏自相关值开始下降，但仍然有一些滞后期的偏自相关值保持在显著水平（例如0.10和0.08），这可能表明原始序列具有一些短期的自相关性。
# 
# 2. **一阶差分后的序列（1st Order Differencing）**:
#    - 一阶差分后的偏自相关图显示，在第一个滞后（lag=1）偏自相关值显著下降，接近0.5，这表明一阶差分后的序列仍然存在一些自相关性。
#    - 随着滞后的进一步增加，偏自相关值迅速下降并接近-0.01和-0.5，这可能表明序列在一阶差分后仍需要进一步处理。
# 
# 3. **二阶差分后的序列（2nd Order Differencing）**:
#    - 二阶差分后的偏自相关图显示，在第一个滞后偏自相关值略有下降，但仍然接近0.5，表明序列可能仍存在一些自相关性。
#    - 然而，从文本中可以看出，随着滞后的增加，偏自相关值迅速下降并接近零或负值，这可能表明二阶差分后的序列自相关性已经大大减弱。
# 
# **结论**:
# - 原始序列的偏自相关图表明序列可能需要差分以减少自相关性。
# - 一阶差分后的序列仍然显示出一些自相关性，这可能表明需要进一步的差分。
# - 二阶差分后的序列偏自相关图显示自相关性已经减弱，这可能表明序列已经接近平稳，或者至少自相关性已经降低到可以接受的水平。
# 
# 请注意，选择正确的差分阶数应该基于综合考虑ACF和PACF图的结果，以及ADF检验的统计量和p值。此外，偏自相关图的截断（即在滞后大于某个值之后偏自相关迅速下降到零）通常表明可以考虑在这个滞后阶数的AR模型。然而，最终模型的选择还应考虑其他因素，如模型的AIC或BIC值，以及模型的诊断检验结果。
# 

# %% [markdown]
# 正如我们通过PACF图来确定AR项的数量一样，我们也会通过ACF图来确定MA项的数量。从技术上讲，MA项是滞后预测的误差。
# 
# ACF显示了为了在平稳系列中消除任何自相关性需要多少MA项。
# 
# 让我们来看一下差分系列的自相关图。
# 
# 

# %%
adf_p_vals

# %%
diff_1_df = train_df[::-1].diff().dropna()
ADF_P_VALs_1 = {}
for col in diff_1_df.columns:
    result = adfuller(diff_1_df.loc[:, col].values)
    ADF_P_VALs_1[col] = result
adf_p_vals_1 = pd.DataFrame(ADF_P_VALs_1, [
    'ADF Statistic', 'p-value', 'Lags Used', 
    'Number of Observations Used', 
    'Critical Values (1%)(5%)(10%)', 
    'icbest'
])
adf_p_vals_1

# %%
diff_2_df = train_df[::-1].diff().diff().dropna()
ADF_P_VALs_2 = {}
for col in diff_2_df.columns:
    result = adfuller(diff_2_df.loc[:, col].values)
    ADF_P_VALs_2[col] = result
adf_p_vals_2 = pd.DataFrame(ADF_P_VALs_2, [
    'ADF Statistic', 'p-value', 'Lags Used', 
    'Number of Observations Used', 
    'Critical Values (1%)(5%)(10%)', 
    'icbest'
])
adf_p_vals_2

# %%
diff_2_df.head()

# %%
from statsmodels.tsa.vector_ar.var_model import VAR

# %%
# model = VAR(train_df[::-1].values)
model = VAR(diff_1_df)
result = model.fit(1)

res = result.forecast(train_df[::-1].values, 3)

first = 2023
predicted_df = pd.DataFrame(res, columns=train_df.columns, index=range(first, first + len(res)))
res_df = pd.concat([train_df[::-1], predicted_df])
index = train_df.index[::-1].append(predicted_df.index)
res_df = pd.DataFrame(std.inverse_transform(res_df), index=index, columns=train_df.columns)
res_df

# %%
plt.figure(figsize=(12, 6), dpi=300)
for col in res_df.columns:
    y = res_df.loc[:, col].values.reshape(-1, 1)
    # print(col, y, y.index)
    plt.plot(data=y, marker='o', label=col)

plt.legend(loc='lower right', bbox_to_anchor=(0.96, -0.2), ncol=10)

plt.title('不同年龄段人口20年内变化情况')
plt.xlabel('年份')
plt.ylabel('标准化后人口数分布情况')

plt.tight_layout()

plt.show()


