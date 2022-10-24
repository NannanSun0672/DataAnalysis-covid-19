"""
Create on Feb 20,2020

@author:nannan.sun

Function:

1.合并数据集

2.对数据进行预处理

"""
import  pandas as pd
import collections
import numpy  as np
from matplotlib import pyplot as plt
#import seaborn as sns
from scipy.stats import pearsonr
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import preprocessing
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
class Data_Process(object):
    def __init__(self):
        self.data_ori_path = "./Data/原始数据 - 副本.xlsx"
        self.data_scored_path = "./Data/总评分表 - 副本.xlsx"
        self.processed_data_path = ""./Data/原始-简化特征数据.xlsx"
        #self.df = self.data_read(self.data_scored_path)
        #self.feature_search()
        self.data_CT_path ="/Users/sunnannan/Desktop/原始无编号-简化特征数据_2.xlsx"
        self.variables_path = "/Users/sunnannan/Desktop/人工肝/4_variables.xlsx"
        self.vif_data_path = "/Users/sunnannan/Desktop/410_New_Info_used.xlsx"
        self.train_path = "./Data/80_percent_train_data.xlsx"
        self.dev_path = "./Data/20_percent_dev_data.xlsx"
        self.ori_Data = "./Data/new_features_data.xlsx"
        #self.test_heatmap = "./Data/test_heatmap.xlsx"
        self.data_R_heatmap = "./Data/data_median_heatmap.xlsx"
        #self.vif_statistic()
        self.variables_importance()
        #self.CT_data_process()
        #self.data_anylisis()
        #self.feature_attibution()
        #self.test_relation()
        #self.data_statistic()
        #self.Data_process()
        #self.heat_map()
    def Z_score(self,data):
        """
        1.
        :return:
        """
        lenth = len(data)
        total = sum(data)
        ave = float(total) / lenth
        #print(ave)
        tempsum = sum([pow(data[i] - ave, 2) for i in range(lenth)])
        tempsum = pow(float(tempsum) / lenth, 0.5)
        for i in range(lenth):
            data[i] = (data[i] - ave) / tempsum
        return data
    def Data_process(self):
        Data = pd.read_excel(self.ori_Data)
        columns = Data.columns.values.tolist()
        Data_pos = Data[Data["诊断"]==1]
        Data_neg = Data[Data["诊断"]==0]
        #print(Data_pos.shape,Data_neg.shape)
        mean_dict = {}
        pos_dict = {}
        neg_dict = {}
        relation_dict = {}
        standard_dict = {}
        for idx, column in enumerate(columns):
            if idx >=0 and column != "诊断":
                #print(list((Data[column] - np.mean(Data[column])) / np.std(Data[column])))
                data = list(Data[column])
                #print(column,data)
                data_standard = self.Z_score(data)
                standard_dict.update({column:data_standard})
        #print(standard_dict)
        standard_dict.update({"诊断":list(Data["诊断"])})
        #print(standard_dict["诊断"])
        standard_Frame = pd.DataFrame(standard_dict)
        data_pos = standard_Frame[standard_Frame["诊断"] == 1]
        data_neg = standard_Frame[standard_Frame["诊断"]==0]
        median_dict = dict()
        for idx,column in enumerate(columns):
            if idx >=0 and column != "诊断":
                pos_list = list(data_pos[column])
                #print(len(pos_list))
                neg_list = list(data_neg[column])
                #print(len(neg_list))
                pos_median = self.get_median(pos_list)
                neg_median = self.get_median(neg_list)
                median_dict.update({column:[pos_median,neg_median]})
        print(median_dict)
        median_data = pd.DataFrame(median_dict)
        median_data.to_excel("/Users/sunnannan/Desktop/data_median_heatmap.xlsx")

    def get_median(self,data):
        """
        取中位数
        :param data:
        :return:
        """
        data.sort()

        half = len(data) // 2

        return (data[half] + data[~half]) / 2

        #Data_means.to_excel("/Users/sunnannan/Desktop/data_means_heatmap.xlsx")
        #####计算相关性

    def heat_map(self):
        """
        1.绘制热力图
        :return:
        """
        Data = pd.read_excel(self.data_R_heatmap)
        Data.set_index('Diagnose',inplace = True)
        ylabels = list(Data.index)

        labels = Data.columns.values.tolist()
        xlabels = list(labels)

        ss = preprocessing.StandardScaler()  # 归一化
        data = ss.fit_transform(Data)

        df = pd.DataFrame(data)

        dfData = df.corr()
        plt.subplots(figsize=(10, 10))  # 设置画面大小
        sns.heatmap(Data, annot=False, vmax=0.5, square=True, yticklabels=ylabels, xticklabels=xlabels, cmap="RdBu")
        plt.show()

    def data_statistic(self):
        """
        1.统计数据分布情况
        :return:
        """
        data = pd.read_excel(self.dev_path)
        #print(data)
        y = data["诊断"]
        val_0_age = list()
        val_1_age = list()
        for idx, class_value in enumerate(list(y)):
            if class_value == 0:
                val_0_age.append(data["CRP(数值)"][idx])
            elif class_value == 1:
                val_1_age.append(data["CRP(数值)"][idx])
        print("阴性患者数量:",len(val_0_age))
        print("阳性患者数量:",len(val_1_age))
        mean_age_0 = np.mean(val_0_age)
        std_age_0 = np.std(val_0_age)
        print("测试集上阴性患者年龄分布", mean_age_0, std_age_0)
        mean_age_1 = np.mean(val_1_age)
        std_age_1 = np.std(val_1_age)
        print("测试集上阳性患者年龄分布", mean_age_1, std_age_1)
        data_dict = data.to_dict("list")

        for key,value_list in data_dict.items():
            #print(key)
            #print(value_list)
            value_1 = []
            value_0 = []
            for idx,index in enumerate(y):
                if int(index)==0:
                    value_0.append(value_list[idx])
                elif int(index)==1:
                    value_1.append(value_list[idx])
            print(key+"阴性患者分布:",Counter(value_0))
            print(key+"阳性患者分布:",Counter(value_1))

    def vif_statistic(self):
        """
        1.计算方差膨胀因子
        :return:
        """
        data = pd.read_excel(self.vif_data_path)
        #print(data)
        #features = (data.columns - ["诊断"])
        #print((features))

        feature_selection = ["年龄方差", "Total ALS performed","BUN (mmol/L)（第一次ALS之前）","TBIL (umol/L)第一次ALS之后"]

        data_2 = data[feature_selection]
        min_max_scaler = preprocessing.StandardScaler()
        info_scaler = min_max_scaler.fit_transform(data_2)
        X = np.matrix(info_scaler)

        #print(data_2)
        # get y and X dataframes based on this regression:
        #y, X = dmatrices('annual_inc ~' + features, df, return_type='dataframe')
        vif = pd.DataFrame()
        #for i in range(data_new.shape[1]):
        #    print(i)

        vif["VIF Factor"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        vif["features"] = data_2.columns
        vif_df = vif.round(2)
        print(vif_df)
        vif_dict = vif_df.to_dict("list")
        print(vif_dict)
        T_dict = {"Tolerance":[],"features":[]}
        for idx,vif in enumerate(vif_dict["VIF Factor"]):
            t = round(1/vif,2)
            T_dict["Tolerance"].append(t)
            T_dict["features"].append(vif_dict["features"][idx])
        print(T_dict)
        vif_dict.update({"Tolerance":T_dict["Tolerance"]})
        print(vif_dict)
        data_t_vif = pd.DataFrame(vif_dict)
        data_t_vif.to_excel("/Users/sunnannan/Desktop/data_t_vif.xlsx")


    def variables_importance(self):
        """
        1.变量重要性直方图
        :return:
        """
        data_frame = pd.read_excel(self.variables_path)
        #print(data_frame)
        data_dict = data_frame.to_dict("list")
        print(data_dict)
        #print(sum(data_dict["变量重要性"]))
        max_num = max(data_dict["变量重要性"])
        print(max_num)
        #importance_list = [round(i/max_num*100) for i in data_dict["变量重要性"]]
        importance_list = [round(i,4) for i in data_dict["变量重要性"]]
        print(importance_list)
        print(data_dict["变量名称"])

        #plt.barh(data_dict["变量名称"],importance_list, height=0.5)
        #plt.bar(data_dict["变量名称"],importance_list)
        plt.bar(data_dict["变量名称"], importance_list)
        #for y, x in enumerate(importance_list):
        #    plt.text(x, y-0.1, "%s" % x)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        for x, y in enumerate(importance_list):
            print(x,y)
            plt.text(x, y , '%.4f' % y, ha='center', va='bottom')
        plt.xlabel("Variable Name",fontdict = {"size":12})
        plt.ylabel("Random forest features importance",fontdict = {"size":16})
        #plt.xlim(1,10000)
        plt.show()
    def test_relation(self):
        """
        1.检验检查特征之间的相关性分析

        :return:
        """
        data_frame = pd.read_excel(self.data_CT_path)
        #print(data_frame)
        wbc = data_frame["白细胞计数(数值)"]
        zhongxingli = data_frame["中性粒细胞计数(数值)"]
        linba = data_frame["淋巴细胞计数(数值)"]
        CRP = data_frame["CRP(数值)"]
        print(list(wbc))
        wbc_list = list(wbc)
        zhong_list = list(zhongxingli)
        linba_list = list(linba)
        CRP_list = list(CRP)


        plt.scatter(wbc_list,zhong_list)
        plt.annotate("r=0.65",xy=(20, 66),fontsize=15)
        plt.annotate("p<0.001",xy=(20,62),fontsize=15)


        #plt.legend()
        plt.xlabel("White Blood Cell (×109/L)")
        plt.ylabel("Neutrophil Cell (×109/L)")

        plt.show()
        np.set_printoptions(suppress=True)
        print(round(pearsonr(wbc_list, zhong_list)[0],5))
        print(round(pearsonr(wbc_list, zhong_list)[1],120))



    def feature_attibution(self):
        """
        绘制CRP、中性粒细胞、淋巴细胞、白细胞数值特征分布
        :return:
        """
        data_frame = pd.read_excel(self.data_CT_path)
        sex_replace = {"女": 0, "男": 1}
        dignosis_replace = {"阴性": 0, "阳性": 1, "弱阳性": 1, "弱阴性": 0, "疑似": 1}
        df = data_frame.replace({"性别": sex_replace, "诊断": dignosis_replace})
        data_CRP = df["CRP(数值)"]
        print(data_CRP)
        print(collections.Counter(data_CRP))
        #fig, ax = plt.subplots()
        #plt.hist(data_CRP,normed=True)

        #plt.xlabel("CRF_value")
        #plt.ylabel("count of samples")
        #plt.show()
        #绘制密度图
        #data_CRP.plot(kind="kde")
        #plt.xlabel("CRF_value")
        #plt.ylabel("density")
        #plt.show()
        #绘制密度图与诊断有关
        #sns.barplot(x="CRP(数值)",y="诊断",data= df)

        facet = sns.FacetGrid(df, hue="诊断", aspect=2)
        facet.map(sns.kdeplot, "中性粒细胞计数(数值)", shade=True)
        #facet.set(xlim=(0, 400))
        facet.set(ylim=(0, 0.3))
        plt.legend()
        plt.xlabel("Numerical value of Neutrophils")
        plt.ylabel('Density distribution')
        plt.show()

    def data_anylisis(self):
        df = pd.read_excel(self.data_CT_path)
        data_dict = df.to_dict("list")
        print("最小年龄为:",min(data_dict["年龄"]))
        print("最大年龄为:",max(data_dict["年龄"]))
        print(data_dict["年龄"])
        print("平均年龄",np.mean(data_dict["年龄"]))
        print(data_dict)
        print(collections.Counter(data_dict["性别"]))

    def CT_data_process(self):
        df = pd.read_excel(self.data_CT_path)
        print(df.shape)
        #CT_replace = {0.0:0,0.3:1}
        #df = df.replace({"CT/胸片X线":CT_replace})
        #print(df)
        df_dict = df.to_dict("list")
        print(df_dict["CT/胸片X线"])
        CT_list = list()
        for idx,value in enumerate(df_dict["CT/胸片X线"]):
            if value == 0:
                CT_list.append(0)
            elif value == 0.3:
                CT_list.append(1)
            elif value == 0.5:
                CT_list.append(2)
            elif value == 1:
                CT_list.append(3)
            elif value == 2:
                CT_list.append(4)
        print(len(CT_list))
        df_dict.update({"CT/胸片X线":CT_list})
        print(df_dict["CT/胸片X线"])
        data_frame = pd.DataFrame(df_dict)
        #data_frame.to_excel("/Users/sunnannan/Desktop/原始无编号-简化特征数据_2.xlsx")


    def data_read(self,data_path):
        """
        1.读取文件数据
        :return:
        """
        data_name = data_path.split("-")[0].split("/")[-1]
        df = pd.read_excel(data_path)
        print(data_name,df.shape)
        features = df.columns.tolist()
        print("数据特征有:",features)
        print('Number of columns: ' + str(len(df.columns)))
        if "原始数据" in data_name:
            #print(df.head(10))
            df = df.drop(columns=["体温（分类）","白细胞计数总数（3.5-9.5）","淋巴细胞计数或淋巴细胞百分比（1.1-3.2/20-50）",
                                  "中性粒细胞计数或中性粒细胞百分比（1.8-6.3/40-75","超敏C反应蛋白  （0-10）","淋巴细胞百分比（数值）","中性粒细胞百分比（数值）","入观/初诊日期"])

            df_new = df[df["诊断"] !="缺失"]
            return df_new

        else:
            return df

    def feature_search(self):
        """
        1.查询特征分布
        :return:
        """
        print(self.df.shape)
        print(self.df.dtypes.unique())
        numeric_dtypes = ['int64', 'float64']
        categorical_dtypes = ['object']
        numeric_columns = self.df.select_dtypes(include=numeric_dtypes).columns.tolist()
        categorical_columns = self.df.select_dtypes(include=categorical_dtypes).columns.tolist()
        print('numeric columns: ' + str(numeric_columns))
        print("counts of numeric columns",len(numeric_columns))
        print('categorical columns: ' + str(categorical_columns))
        print("counts of categorical columns",len(categorical_columns))
        #数据特征分布
        features_attri = self.df[numeric_columns].describe()
        #print("数据特征分布\n",features_attri)
        #features_attri.to_excel("/Users/sunnannan/Desktop/features_attri.xlsx")
        #查看缺失值
        data_shorted = self.df.isna().sum()
        #填充缺失值,哑变量用0填充，连续变量按照"诊断"划分数据后对该变量进行均值填充
        #self.df['有无旅行史或居住史（湖北其他城市）'] = self.df['有发热或呼吸道症状患者接触史（湖北其他城市）'].fillna(0)
        print(data_shorted)
        df_new = self.df.drop(columns=["编号"])
        df_new2 = df_new.drop(columns=["入观/初诊日期"])
        df_new2.to_excel("./Data/data.xlsx")
if __name__ == "__main__":
    Data_Process = Data_Process()
