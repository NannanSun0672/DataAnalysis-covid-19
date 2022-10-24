"""
Create on Feb 18,2020

@author:nannan.sun@

Function:


1.进行数据处理，转换

2.建立决策树模型

3.数据预测

"""

import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,LeaveOneOut,GridSearchCV, KFold,cross_val_score
from sklearn.preprocessing import scale
from sklearn.metrics import mean_absolute_error, accuracy_score,confusion_matrix,classification_report,cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import feature_selection
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing
import  matplotlib
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
#from xgboost import XGBClassifier
from collections import Counter
#import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.utils import shuffle
#import statsmodels.api as sm
from sklearn.feature_selection import VarianceThreshold,chi2,f_classif,SelectKBest


class Model_classification(object):
    def __init__(self):

       
        self.data_path = "./Data/data.xlsx" ###中华约稿

        self.trans_df,self.feature = self.Data_import()
        self.trans_df = self.trans_df.dropna()
        print(self.trans_df.shape)
        #self.trans_df_786 = self.trans_df[:786]
        #print(self.trans_df_786)
        ####绘制条形图
        #self.hist_plot()

        self.columns = list(self.feature)
        print("原始数据特征数量",len(self.columns))
        self.columns.remove("诊断")
        #self.feature_selection = ["CT/胸片X线","家中或工作单位有无聚集性发病史",
        #                          "有无旅行史或居住史(武汉)","有发热或呼吸道症状患者接触史(武汉)","有发热或呼吸道症状患者接触史(其他本地病例持续增加地区)",
        #                          "肌酸肌痛","呼吸困难","乏力","淋巴细胞计数(数值)","白细胞计数(数值)"]
        #self.feature_selection = ["CT/胸片X线","肌酸肌痛","呼吸困难","乏力","淋巴细胞计数(数值)","白细胞计数(数值)"]
        #self.feature_selection = ["CT/X RAY","聚集性发病","居旅史（合并）","接触史（合并）","肌肉酸痛","呼吸困难","乏力","鼻塞流涕","L","WBC"]

        #self.test_path = "/Users/sunnannan/Desktop/居总数据_处理6.xlsx"#测试集路径
        #self.test_data = pd.read_excel(self.test_path)
        #print(self.test_data.shape)
        #self.recons_data = self.reconstruction_Data()###中华约稿
        #self.recons_features = self.recons_data.columns.tolist()###中华约稿
        #self.recons_features.remove("诊断")###中华约稿
        self.feature_selection = ["有无确诊患者接触史","乏力","呼吸困难","肌酸肌痛","年龄","白细胞计数(数值)",
                                   "淋巴细胞计数(数值)","中性粒细胞计数(数值)","NLR"]
        self.min_max_scaler = preprocessing.StandardScaler()
        self.X,self.y,self.X_train, self.X_val, self.y_train, self.y_val = self.date_prepare()
        #print(self.X)
        #self.feature_selection()


        #self.columns = list(self.X_train.to_dict("list").keys())
        #print(self.columns)
        #self.test.to_excel("/Users/sunnannan/Desktop/测试数据.xlsx")
        #print(self.y_test)
        #self.max_leaf_nodes = [self.get_mae(x) for x in [5, 50, 500, 5000]]
        #print(self.max_leaf_nodes)
        #self.fpr_DT, self.tpr_DT,self.pre_y_DT = self.DecisonTree_Model()



        self.fpr_RF, self.tpr_RF,self.pre_y_RF = self.RandomForest_Model()
        self.fpr_LR, self.tpr_LR = self.logistic_Model()
        self.fpr_SVM,self.tpr_SVM = self.SVM_Model()
        self.fpr_DNN,self.tpr_DNN = self.DNN_Model()
        #self.load_Model()
        self.plot_ROC()
        #print("决策树预测结果",self.pre_y_DT)
        #print("随机森林模型预测结果",self.pre_y_RF)
        #kappa = cohen_kappa_score(self.pre_y_RF,self.pre_y_DT,labels=None, weights=None, sample_weight=None)
        #print("kappa",kappa)
    def feature_selection(self):
        """
        1. 方差阈值化
        2.相关矩阵检查
        2. 卡方检验，方差分析F值
        :return:
        """
        Categorical_Variable = ["性别","有无旅行史或居住史(疫情爆发区)","有无确诊患者接触史","干咳","乏力","呼吸困难","肌酸肌痛"]
        Numerical_Variable = ["年龄","体温（数值℃）","白细胞计数(数值)","淋巴细胞计数(数值)","中性粒细胞计数(数值)","淋巴细胞百分比（数值）",
                              "中性粒细胞百分比（数值）","CRP(数值)","NLR"]
        ######方差阈值化
        Categorical_data = self.trans_df[Categorical_Variable]
        #print(np.array(Categorical_data))
        Numerical_data = self.trans_df[Numerical_Variable]
        Numerical_threshold = VarianceThreshold(threshold=0.5)
        C_high_features = Numerical_threshold.fit(Numerical_data)
        #print("方差系数",dict(zip(Numerical_Variable,C_high_features.variances_)))
        ######相关矩阵
        corr_matrix = self.trans_df.corr().abs()
        print(corr_matrix)
        #upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))
        #to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        ######卡方检验，方差分析F值
        for idx, feature in enumerate(Categorical_Variable):
            chi2_,p_value = chi2(np.array(self.trans_df[[feature]]),np.array(self.trans_df["诊断"]))
            print(feature,chi2_[0],p_value[0])
        #chi2_selection = SelectKBest(chi2,k = 4)
        #feature_kbest = chi2_selection.fit_transform(self.trans_df[Categorical_Variable],self.trans_df["诊断"])
        #print(feature_kbest)
        for idx,feature in enumerate(Numerical_Variable):
            f_, p_value = f_classif(np.array(self.trans_df[[feature]]), np.array(self.trans_df["诊断"]))
            print(feature, f_[0], p_value[0])
        #print(self.min_max_scaler(self.trans_df.values()))

    def plot_ROC(self):
        """
        1.绘制多条ROC曲线
        :return:
        """

        #plt.figure()
        figure,ax = plt.subplots()
        lw = 2
        plt.figure(figsize=(10, 10))
        #plt.plot(self.fpr_DT, self.tpr_DT, color='blue', lw=lw)  ###假正率为横坐标，真正率为纵坐标做曲线

        plt.plot(self.fpr_RF, self.tpr_RF, color='black', lw=lw)

        plt.plot(self.fpr_LR, self.tpr_LR, color='red', lw=lw)

        plt.plot(self.fpr_SVM, self.tpr_SVM, color='dodgerblue', lw=lw)

        plt.plot(self.fpr_DNN,self.tpr_DNN, color ='violet',lw=lw)

        #plt.xlim([0.0, 1.0])
        #plt.ylim([0.0, 1.05])
        ###设置刻度值字体大小
        plt.tick_params(labelsize=20)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 30,
                 }
        plt.ylabel('Sensitivity',font2)
        plt.xlabel('1-Specificity',font2)

        plt.title('Receiver Operating Characteristic',font2)
        #plt.legend(loc="lower right")
        #plt.savefig(base + "roc_img\\sobel_roc.jpg")
        plt.show()

    def hist_plot(self):
        #print(self.trans_df.groupby(["诊断"])['结膜充血  '])

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        df_dict = self.trans_df.to_dict("list")
        df_dignose = df_dict["诊断"]
        #print(len(df_dignose))
        wcell_list = df_dict['咽痛咽干']
        #print(len(wcell_list))
        c = Counter(wcell_list)
        #print(c)
        x_label = ["0","1"]
        x = np.arange(len(x_label))
        y_0_0 = []
        y_0_1 = []
        y_1_0 = []
        y_1_1 = []
        for idx,item in enumerate(df_dignose):
            if item == 0:
                if wcell_list[idx] == 0:
                    y_0_0.append(wcell_list[idx])
                elif wcell_list[idx] == 1:
                    y_0_1.append(wcell_list[idx])
            elif item ==1:
                if wcell_list[idx] == 0:
                    y_1_0.append(wcell_list[idx])
                elif wcell_list[idx] == 1:
                    y_1_1.append(wcell_list[idx])
        y_0 = [len(y_0_0),len(y_1_0)]
        y_1 = [len(y_0_1),len(y_1_1)]

        print(len(y_0_1))
        print(len(y_0_0))
        print(len(y_1_1))
        print(len(y_1_0))
        bar_width = 0.3  # 条形宽度
        index_0 = np.arange(len(x_label))  # 男生条形图的横坐标
        index_1 = index_0 + bar_width
        plt.bar(index_0, height=y_0, width=bar_width, color='b', label='0')
        plt.bar(index_1, height=y_1, width=bar_width, color='g', label='1')

        plt.legend()
        plt.xticks(index_0 + bar_width / 2, x_label)
        plt.ylabel('Counts of samples')
        #plt.title('Distribution of conjunctival congestion in positive and negative samples')
        #p1 = plt.bar(x, height=y, width=0.5, label="WBC-Samples count", tick_label=x_label)

        for a, b in zip(x, y_0):
            plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)
        #for a, b in zip(x, y_1):
        #    plt.text(rect.get_x() + rect.get_width() / 2a, b + 0.3, '%.0f' % b, ha='center', va='bottom', fontsize=10)

        #plt.legend()
        plt.show()

    def Data_import(self):
        """
        1.Data import
        2.classify features
        :return:
        """
        data_frame = pd.read_excel(self.data_path)
        print(data_frame.shape)
        #print("查看缺失值",data_frame.isna().sum())
        #数据类型
        #print("数据类型",data_frame.dtypes.unique())
        #numeric_dtypes = ['int64', 'float64']
        #categorical_dtypes = ['object']
        #numeric_columns = data_frame.select_dtypes(include=numeric_dtypes).columns.tolist()
        #categorical_columns = data_frame.select_dtypes(include=categorical_dtypes).columns.tolist()
        #print('numeric columns: ' + str(numeric_columns))
        #print('categorical columns: ' + str(categorical_columns))
        #df = data_frame.drop(columns=["入观/初诊日期","总分"])
        #将非数量数据转化为数量级数据
        #sex_replace = {"女":0,"男":1,"男 ":1}
        #dignosis_replace = {"阴性":0,"阳性":1,"弱阳性":1,"弱阴性":0,"疑似":1}
        #df = data_frame.replace({"性别":sex_replace})
        #print(df["诊断"])
        #将基础疾病转换为{"有"：1,"无":0}
        df_dict = data_frame.to_dict("list")
        #print(df_dict.keys())

        for key,value_list in df_dict.items():

        #    if key == "基础疾病":
        #        base_disease =list()
        #        dis = list()
        #        for value in value_list:
        #            dis.append(value)
        #            if "无" in value:

        #                base_disease.append(0)
         #           else:
         #               base_disease.append(1)
         #       df_dict.update({key:base_disease})
                #print(base_disease)
                #print(dis)
        #print(len(df_dict["基础疾病"]))
            if key == "诊断":
                pos = list()
                neg = list()
                for value in value_list:
                    if value == 1:
                        pos.append(1)
                    else:
                        neg.append(0)
                print("阳性患者病例数:",len(pos))
                print("阴性患者病例数:",len(neg))

        df_transformed = pd.DataFrame(df_dict)
        #print(df_transformed.shape)
        #print("数值转换过数据缺失情况\n",df_transformed.isna().sum())
        #df_transformed.to_excel("/Users/sunnannan/Desktop/trans_data.xlsx")
        return df_transformed,df_dict.keys()
    def date_prepare(self):
        """
        1.选择
        2.数据归一化
        3.随机选择训练集和测试集，20%测试集合
        :return:
        """
        features = list(self.feature_selection)
        #test_Data = self.recons_data###中华约稿
        #features = self.recons_data.columns.tolist()###中华约稿

        #features.remove("诊断")###中华约稿
        print("features", len(features))
        print("features",features)
        X = self.trans_df[features]
        y = self.trans_df["诊断"]

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=101)

        #X_test = self.test_data[features]
        #y_test = self.test_data["诊断"]

        return X,y,X_train, X_val,y_train, y_val
    def reconstruction_Data(self):
        """
        重新构建数据属性
        :return:
        """

        Info_dict = self.trans_df.to_dict("list")
        Res_Info = dict()
        outbreak_history = list()
        contact_history = list()
        for idx, unit in enumerate(Info_dict["有无旅行史或居住史（武汉）"]):
            if unit == 0 and Info_dict["有无旅行史或居住史（湖北其他城市）"][idx] == 0 and Info_dict["有无旅行史或居住史（其他本地病例持续增加地区）"][
                idx] == 0:
                outbreak_history.append(0)
            else:
                outbreak_history.append(1)
            # print(outbreak_history)
        Res_Info.update({"有无旅行史或居住史(疫情爆发区)": outbreak_history})
        for idx, unit in enumerate(Info_dict["有发热或呼吸道症状患者接触史（武汉）"]):
            if unit == 0 and Info_dict["有发热或呼吸道症状患者接触史（湖北其他城市）"][idx] == 0 and \
                        Info_dict["有发热或呼吸道症状患者接触史（其他本地病例持续增加地区）"][idx] == 0 and \
                        Info_dict["家中或工作单位有无聚集性发病史 "][idx] == 0:
                contact_history.append(0)
            else:
                contact_history.append(1)
            # print(contact_history)
        Res_Info.update({"有无确诊患者接触史": contact_history})
        Res_Info.update({"性别": Info_dict["性别"], "年龄": Info_dict["年龄"], "诊断": Info_dict["诊断"], \
                             "体温（数值℃）": Info_dict["体温（数值℃）"], "干咳": Info_dict["干咳   "], "乏力": Info_dict["乏力 "], \
                             "呼吸困难": Info_dict["呼吸困难   "], "肌酸肌痛": Info_dict["肌酸肌痛"], "白细胞计数(数值)": Info_dict["白细胞计数（数值）"], \
                             "淋巴细胞计数(数值)": Info_dict["淋巴细胞计数（数值）"], "中性粒细胞计数(数值)": Info_dict["中性粒细胞计数（数值）"],\
                            "淋巴细胞百分比（数值）":Info_dict["淋巴细胞百分比（数值）"],"中性粒细胞百分比（数值）":Info_dict["中性粒细胞百分比（数值）"],
                             "CRP(数值)": Info_dict["CRP（数值）"],"NLR":Info_dict["NLR"]})

        # print(Res_Info.keys())
        data_frame = pd.DataFrame(Res_Info)
        print("数据总数",data_frame.shape)

        df_pos = data_frame[data_frame["诊断"] == 1]
        df_neg = data_frame[data_frame["诊断"] == 0]
        # print(df_pos)
        # print(df_neg)

        #Info_pos = df_pos.sample(n=178, random_state=1)
        info = df_pos[df_pos["肌酸肌痛"]==1]
        print("info",info.shape)
        # print(Info_pos)
        print(np.mean(df_pos["白细胞计数(数值)"]))
        print(np.std(df_pos["白细胞计数(数值)"]))

        #Info_neg = df_neg.sample(n=272, random_state=1)
        info = df_neg[df_neg["肌酸肌痛"] == 1]
        print("info",info.shape)
        print(np.mean(df_neg["白细胞计数(数值)"]))
        print(np.std(df_neg["白细胞计数(数值)"]))


        data_frame.to_excel("./Data/new_features_data.xlsx")

        return data_frame


    def get_mae(self,max_leaf_nodes):
        """
        1.选择最大叶子结点
        :param max_leaf_nodes:
        :return:
        """
        decision_tree_model = DecisionTreeClassifier(random_state=101, max_leaf_nodes=max_leaf_nodes)

        decision_tree_model.fit(self.X_train, self.y_train)
        predict_tree_default = decision_tree_model.predict(self.X_val)
        return mean_absolute_error(self.y_val, predict_tree_default)

    def DecisonTree_Model(self):
        """
        1.建立决策树模型
        2.模型预测
        3.输出结果
        :return:
        """
        print("............DecisionTree Model................")
        print(self.X_train.shape)
        #print("训练集数据",self.X_train.shape)
        decision_tree_model = DecisionTreeClassifier(random_state=101, criterion="gini",max_leaf_nodes=50)
        decision_tree_model.fit(self.X_train, self.y_train)

        n = decision_tree_model.feature_importances_
        #print("DecisonTree Model important feature \n",n)
        feature_importance = dict(zip(self.feature_selection, n))
        # print(feature_importance)
        feature_sorted = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        print("特征重要性排序（降序）\n", feature_sorted)
        sorted_dict = dict()
        key_list = []
        value_list = []

        for key, value in feature_sorted.items():
            key_list.append(key)
            value_list.append(value)
        sorted_dict.update({"变量名称": key_list, "变量重要性": value_list})
        print(sorted_dict)

        feature_df = pd.DataFrame(sorted_dict)
        feature_df.to_excel("./Data/DT_importance_of_variables.xlsx")
        #feature_df.to_excel("/Users/sunnannan/Desktop/DT_importance_of_variables.xlsx")

        #model = SelectFromModel(decision_tree_model, threshold=0.05,prefit=True)
        #X_new = model.transform(self.X_train)
        #X_val_new = model.transform(self.X_val)
        #print("新选择特征维度:",X_new.shape)

        #留一法
        """
        kfold = KFold(n_splits=len(self.X_train))
        param_grid = {"criterion": ["entropy"]}
        decisionTree_model = DecisionTreeClassifier(random_state=101, criterion="entropy",max_leaf_nodes=50)
        grid = GridSearchCV(decisionTree_model, param_grid, cv=kfold)
        grid.fit(self.X_train, self.y_train.ravel())
        """
        predict_tree = decision_tree_model.predict(self.X_val)
        predict_score = decision_tree_model.predict_proba(self.X_val)
        accuracy = accuracy_score(self.y_val, predict_tree)
        print('DecisonTree model---Test accuracy: {:.4f}'.format(accuracy))
        auc = metrics.roc_auc_score(self.y_val, predict_score[:,1])
        print('DecisonTree model---Test AUC: {:.4f}'.format(auc))
        print("输出验证集预测结果的混淆矩阵\n", confusion_matrix(self.y_val, predict_tree, labels=[0, 1]))  # 输出预测结果的混淆矩阵
        print("-------------------------")
        print("验证集打印分类报告\n", classification_report(self.y_val, predict_tree))  # 打印分类报告
        print("-------------------------")
        fpr, tpr, thresholds = metrics.roc_curve(self.y_val, predict_score[:, 1],drop_intermediate = False)
        #plt.title('Receiver Operating Characteristic')
        #plt.plot(fpr, tpr)
        # plt.annotate("RF-model AUC 0.956", xy=(0.8, 0.2), fontsize=8)
        #plt.plot()
        #plt.ylabel('Sensitivity')
        #plt.xlabel('1-Specificity')
        # plt.legend()
        #plt.show()
        roc_value = dict()
        roc_value.update({"fpr": list(fpr), "tpr": list(tpr), "cutoff_thresholds": list(thresholds)})
        print(roc_value)
        df_roc = pd.DataFrame(roc_value)
        df_roc.to_excel("/Users/sunnannan/Desktop/DT_df_roc.xlsx")

        return fpr,tpr,predict_tree

    def RandomForest_Model(self):
        """
        1.建立随机森林模型
        2.模型预测
        3.输出结果
        :return:
        """
        print("............RandomForest Model................")
        print("训练集样本统计:",self.X_train.shape)

        random_forest_model = RandomForestClassifier(random_state=50, criterion="gini",n_estimators=40)
        #scores = cross_val_score(random_forest_model,self.X_train, self.y_train.ravel(),scoring="roc_auc",cv=10)
        #print(scores)
        #10折交叉验证
        #kfold = KFold(n_splits=10)
        #param_grid = {"n_estimators": [50]}
        #grid = GridSearchCV(random_forest_model, param_grid, cv=kfold)
        #X_new = self.min_max_scaler.fit_transform(self.X_train)
        random_forest_model.fit(self.X_train, self.y_train.ravel())
        #print(grid.cv_results_)
        #pred = grid.predict_proba(self.X_val)
        #auc_grid = metrics.roc_auc_score(self.y_val, pred[:, 1])
        #print("auc",auc_grid)

        n = random_forest_model.feature_importances_
        feature_importance = dict(zip(self.feature_selection, n))
        # print(feature_importance)
        feature_sorted = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        print("升序\n", feature_sorted)
        sorted_dict = dict()
        key_list = []
        value_list = []

        for key,value in feature_sorted.items():
            key_list.append(key)
            value_list.append(value)
        sorted_dict.update({"变量名称":key_list,"变量重要性":value_list})
        print(sorted_dict)

        feature_df = pd.DataFrame(sorted_dict)
        feature_df.to_excel("./Data/RF_importance_of_variables.xlsx")
        #打印训练集
        #train_data_pred = random_forest_model.predict(self.X_train)
        #train_data_score = random_forest_model.predict_proba(self.X_train)

        predict_random_forest = random_forest_model.predict(self.X_val)

        predict_score = random_forest_model.predict_proba(self.X_val)
        print("----------训练集做模型验证--------")
        #accuracy_train = accuracy_score(self.y_train, train_data_pred)
        #print('RandomForest model---Test accuracy: {:.4f}'.format(accuracy_train))
        #print("输出训练集预测结果的混淆矩阵\n", confusion_matrix(self.y_train, train_data_pred, labels=[0, 1]))  # 输出预测结果的混淆矩阵
        #print("-------------------------")
        #print("训练集打印分类报告\n", classification_report(self.y_train, train_data_pred))  # 打印分类报告
        #print("-------------------------")
        #auc_train = metrics.roc_auc_score(self.y_train, train_data_score[:, 1])
        #print('RandomForest model---Train AUC: {:.4f}'.format(auc_train))

        print("-------测试集做模型验证结果--------")

        accuracy = accuracy_score(self.y_val, predict_random_forest)
        print('RandomForest model---Test accuracy: {:.4f}'.format(accuracy))
        print("输出验证集预测结果的混淆矩阵\n", confusion_matrix(self.y_val, predict_random_forest, labels=[0, 1]))  # 输出预测结果的混淆矩阵
        print("-------------------------")
        print("验证集打印分类报告\n", classification_report(self.y_val, predict_random_forest))  # 打印分类报告
        print("-------------------------")
        auc = metrics.roc_auc_score(self.y_val, predict_score[:, 1])
        print('RandomForest model---Test AUC: {:.4f}'.format(auc))
        roc_value = dict()
        fpr, tpr, thresholds = metrics.roc_curve(self.y_val, predict_score[:, 1],drop_intermediate = False)
        roc_value.update({"fpr":list(fpr),"tpr":list(tpr),"cutoff_thresholds":list(thresholds)})
        print(roc_value)
        df_roc = pd.DataFrame(roc_value)
        df_roc.to_excel("/Users/sunnannan/Desktop/RF_df_roc.xlsx")


        #print(len(fpr), len(tpr), len(thresholds))
        # 绘制ROC曲线
        baseline_fpr = []

        #plt.title('Receiver Operating Characteristic')
        #plt.plot(fpr, tpr,color= "dodgerblue")
        #plt.annotate("RF-model AUC 0.956", xy=(0.8, 0.2), fontsize=8)
        #plt.plot()
        #plt.ylabel('Sensitivity')
        #plt.xlabel('1-Specificity')
        #plt.legend()
        #plt.show()
        #AUC = metrics.auc(fpr, tpr)
        #print(AUC)
        return fpr, tpr,predict_random_forest

    def logistic_Model(self):
        """
        1.建立逻辑斯蒂回归模型
        2.模型预测
        3.输出结果
        :return:
        """
        print("............Logistic Model................")
        print("训练集样本统计:", self.X.shape)
        logistic_regression = LogisticRegression(random_state=101, solver='liblinear')
        X_new = self.min_max_scaler.fit_transform(self.X)
        print(X_new.shape)

        logistic_regression.fit(X_new, self.y)
        #summary = logistic_regression.summary()
        #predict_logistic_regression = logistic_regression.predict(self.X_val)
        n = logistic_regression.coef_.ravel()
        print(logistic_regression.intercept_)
        #print("logistic Model\n",n)
        feature_importance = dict(zip(self.feature_selection, n))
        print(feature_importance)
        feature_sorted = dict(sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True))
        print(feature_sorted)
        sorted_dict = dict()
        key_list = []
        value_list = []

        for key, value in feature_sorted.items():
            key_list.append(key)
            value_list.append(abs(value))
        sorted_dict.update({"变量名称": key_list, "变量重要性": value_list})
        print(sorted_dict)

        feature_df = pd.DataFrame(sorted_dict)
        print(feature_df)
        feature_df.to_excel("/Users/sunnannan/Desktop/logit_importance_of_variables.xlsx")
        # 打印训练集


        #feature_df = pd.DataFrame(feature_dict)


        #feature_df.to_excel("/Users/sunnannan/Desktop/LR_features_coef_786.xlsx")

        #print("特征重要性排序（降序）\n",feature_sorted)
        #print("前14个特征\n",feature_sorted.keys)
        #model = SelectFromModel(logistic_regression,threshold=0.25,prefit=True)
        #X_new = model.transform(self.X_train)
        #print(X_new)
        #X_val_new = model.transform(self.X_val)

        #print("新选择特征维度:",X_new.shape)

        # 交叉验证法-----留一法
        kfold = KFold(n_splits=10)
        param_grid = {"C":[1]}
        logistic_regression_model = LogisticRegression(random_state=101,solver='liblinear')

        grid = GridSearchCV(logistic_regression_model,param_grid, cv=kfold)
        pipe = Pipeline([("scaler", self.min_max_scaler), ("logistic", grid)])  # 有两个步骤
        pipe.fit(self.X_train, self.y_train.ravel())
        #joblib.dump(pipe,"/Users/sunnannan/Desktop/NCP/NCP_Pre/app/ie/NCP/model/ncp.pkl")
        predict_score = pipe.predict_proba(self.X_val)
        y_pre = pipe.predict(self.X_val)
        #y_test_pre = pipe.predict(self.test)
        #y_test_score = pipe.predict_proba(self.X_test)
        #输出分类错误样本
        #self.error_classfied_samples("logistic",y_pre)
        accuracy = accuracy_score(self.y_val, y_pre)
        #accuracy_test = accuracy_score(self.y_test, y_test_pre)
        print("-------------------------")
        print('LR model---Val accuracy: {:.4f}'.format(accuracy))
        print("-------------------------")
        print("输出验证集预测结果的混淆矩阵\n",confusion_matrix(self.y_val,y_pre,labels=[0,1]))  # 输出预测结果的混淆矩阵
        print("-------------------------")
        print("验证集打印分类报告\n",classification_report(self.y_val, y_pre))  # 打印分类报告
        print("-------------------------")
        auc = metrics.roc_auc_score(self.y_val, predict_score[:, 1])
        fpr, tpr, thresholds = metrics.roc_curve(self.y_val, predict_score[:, 1],drop_intermediate = False)
        #print(fpr, tpr, thresholds)
        #绘制ROC曲线
        #plt.title('Receiver Operating Characteristic')
        #plt.plot(fpr, tpr,color="blue")
        #plt.ylabel('Sensitivity')
        #plt.xlabel('1-Specificity')

        #plt.show()
        #验证ROC曲线
        #AUC = metrics.auc(fpr, tpr)
        #print(AUC)

        print('LR model---Val AUC: {:.4f}'.format(auc))
        #print('Logistic model---Test accuracy: {:.4f}'.format(accuracy_test))
        #print("输出验证集预测结果的混淆矩阵\n", confusion_matrix(self.y_test, y_test_pre, labels=[0, 1]))  # 输出预测结果的混淆矩阵
        #print("验证集打印分类报告\n", classification_report(self.y_test, y_test_pre))  # 打印分类报告
        #auc_test = metrics.roc_auc_score(self.y_test, y_test_score[:, 1])
        #print('Logistic model---Test AUC: {:.4f}'.format(auc_test))
        print("LR-fpr:",fpr)
        print("LR-tpr:",tpr)
        return fpr, tpr



    def SVM_Model(self):
        """
        1.建立支持向量机回归模型
        2.模型预测
        3.输出结果
        :return:
        """
        print("训练集样本统计:", self.X_train.shape)
        svm_model = SVC(kernel='rbf',probability=True,C=10,gamma=0.001,decision_function_shape='ovr')
        svm_model.fit(self.X_train, self.y_train.ravel())
        #print(svm_model.n_support_)
        predict_svm = svm_model.predict_proba(self.X_val)
        #accuracy = accuracy_score(self.y_val, predict_svm)
        #print('Test accuracy: {:.4f}'.format(accuracy))

        #print("输出验证集预测结果的混淆矩阵\n", confusion_matrix(self.y_val, predict_svm, labels=[0, 1]))  # 输出预测结果的混淆矩阵
        print("-------------------------")
        #print("验证集打印分类报告\n", classification_report(self.y_val, predict_svm))  # 打印分类报告
        #print("-------------------------")

        auc = metrics.roc_auc_score(self.y_val, predict_svm[:, 1])

        print('SVM model---Test AUC: {:.4f}'.format(auc))
        fpr, tpr, thresholds = metrics.roc_curve(self.y_val, predict_svm[:, 1],drop_intermediate = False)
        #print(fpr, tpr, thresholds)
        # 绘制ROC曲线
        #plt.title('Receiver Operating Characteristic')
        #plt.plot(fpr, tpr,color="red")
        #plt.ylabel('Sensitivity')
        #plt.xlabel('1-Specificity')
        #plt.show()
        #fpr = 0
        #tpr = 0
        return fpr, tpr

    def xgboost_Model(self):
        """
        1.建立xgboost 模型
        2.模型预测
        3.输出结果
        :return:
        """

    def get_fc_model(self):
        model = Sequential();
        model.add(Dense(64, input_shape=(9,), activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        return model
    def DNN_Model(self):
        """
        1.建立DNN模型
        2.模型预测
        3.输出结果
        :return:
        """
        fc_model = self.get_fc_model()
        fc_model.compile(optimizer='Adam', loss='mean_squared_error',
                         metrics=['accuracy'])
        fc_model.optimizer.lr = 0.01
        fc_model.fit(x=self.X_train.values, y=self.y_train.values, epochs=100)
        predictions = fc_model.predict_proba(self.X_val, verbose=0)
        y_pre = fc_model.predict_classes(self.X_val,verbose=0)
        #print(len(predictions))
        #predictions = predictions.reshape(1, 183)
        #predictions = predictions[0]
        #print(predictions)
        fpr_keras, tpr_keras, thresholds_keras = metrics.roc_curve(self.y_val, predictions,drop_intermediate = False)
        #print(fpr_keras,tpr_keras,thresholds_keras)
        # 计算 AUC
        AUC = metrics.auc(fpr_keras, tpr_keras)

        print("DNN Model-AUC",AUC)
        print("DNN Model 输出验证集预测结果的混淆矩阵\n", confusion_matrix(self.y_val, y_pre, labels=[0, 1]))

        return fpr_keras,tpr_keras


    #def load_Model(self):
    #    """
    #    1.加载模型
    #    :return:
    #    """
        #lr = joblib.load("/Users/sunnannan/Desktop/ncp.pkl")
        # 进行模型的预测
        #y_pred = lr.predict(self.X_test)
        #accuracy_test = accuracy_score(self.y_test, y_pred)
        #print(accuracy_test)

if __name__ == "__main__":
    classificator = Model_classification()



