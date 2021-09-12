from lib import *

class Data:
    def __init__(self, fold, Xcols, get_Xm=False, seed=None):
        self.Xscaler = StandardScaler()
        self.yscaler = StandardScaler()
        self.fold = fold
        self.Xcols = Xcols.split('@')
        #######
        self.suffix = 'mar_nsgp'
        self.factor = 4 # Factor of data for Xm
        self.start = '2015-03-01'
        self.end = '2015-03-31'
        self.get_Xm = get_Xm
        self.seed = seed
        self.all_cont_cols = ['longitude', 'latitude', 'temperature', 'humidity', 'wind_speed', 'delta_t']
        # self.cat_factor = 100

    def common(self, data):
        data['time'] = pd.to_datetime(data['time'])
        data = data.set_index('time')
        data = data[self.start:self.end]
        X = data[self.Xcols]
        y = data[['PM25_Concentration']]
        return X, y, data

    def load_train(self):
        train_data = pd.read_csv(Config.data_path+'fold'+self.fold+'/train_data_'+self.suffix+'.csv.gz')
        # print(train_data.shape)
        X, y, self.train_data = self.common(train_data)
        # print(X.shape, y.shape)
        
        self.cont_cols = [i for i in self.Xcols if i in self.all_cont_cols]
        X[self.cont_cols] = self.Xscaler.fit_transform(X[self.cont_cols])

        ###### 
        self.cat_indicator = [0 if i in self.cont_cols else 1 for i in X.columns]
        self.time_indicator = [1 if i in ['delta_t'] else 0 for i in X.columns]

        # print(X.iloc[0])
        # y[y.columns] = self.yscaler.fit_transform(y)
        # Or
        self.y_mean = y.values.mean();y = y - self.y_mean
        self.Xcols = X.columns
        print(X.head(5))
        X = torch.tensor(X.values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.float32)
        if not self.get_Xm:
            return X, y, self.Xcols
        else:
            tindex = self.train_data.index.unique()
            # print(self.train_data.shape, self.train_data.columns)
            # Xm = self.train_data.loc[tindex[::self.factor], Xcols]
            # Or
            Xm = self.train_data.sample(self.train_data.shape[0]//self.factor, random_state=self.seed)[self.Xcols]
            Xm[Xm.columns] = self.Xscaler.transform(Xm)
            Xm = torch.tensor(Xm.values, dtype=torch.float32)
            return X, y, Xm

    def load_test(self):
        test_data = pd.read_csv(Config.data_path+'fold'+self.fold+'/test_data_'+self.suffix+'.csv.gz')
        test_output = pd.read_csv(Config.data_path+'fold'+self.fold+'/test_output_'+self.suffix+'.csv.gz')

        test_data['PM25_Concentration'] = test_output['PM25_Concentration'].values

        X, y, self.test_data = self.common(test_data)

        X[self.cont_cols] = self.Xscaler.transform(X[self.cont_cols])
        # print(X.head(5))
        # X[self.cat_cols] = X[self.cat_cols] - 0.5
        # print(X.head(5), self.cat_factor)
        # X[self.cat_cols] = X[self.cat_cols]*self.cat_factor
        # print(X.head(5))
        # y[y.columns] = self.yscaler.transform(y)

        X = torch.tensor(X.values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.float32)

        return X, y