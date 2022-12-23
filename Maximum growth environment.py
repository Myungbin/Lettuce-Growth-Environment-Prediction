# CTGAN
import os
import warnings

import pandas as pd
import numpy as np
from sdv.tabular import CTGAN
from sdv.sampling import Condition
from autogluon.tabular import TabularDataset, TabularPredictor

from feature.preprocessing import preprocess_day0, preprocess_day_other
from feature.gan_preprocessing import make_traindata_ctgan, make_raw
from config import Settings
warnings.filterwarnings('ignore')


class CreateGE:
    def __init__(self):
        self.gan_model = CTGAN(verbose=True, epochs=500, cuda=True)
        self.predict_model = TabularPredictor.load(
            Settings.predict_model_path, require_version_match=False)

    @property
    def generated_data(self):
        """CTGAN을 이용한 데이터 생성

        Returns:
            100 sample generated data 
        """
        print("Start generated_data")

        for i in range(28):
            print(f"=======Start Day{i}===========")

            raw_data1 = make_traindata_ctgan(Settings.case11)
            raw_day1 = raw_data1.iloc[i:i + 1]

            raw_data2 = make_traindata_ctgan(Settings.case13)
            raw_day2 = raw_data2.iloc[i:i + 1]

            raw_data3 = make_traindata_ctgan(Settings.case14)
            raw_day3 = raw_data3.iloc[i:i + 1]

            raw_day = pd.concat([raw_day1, raw_day2, raw_day3], axis=0)

            self.gan_model.fit(raw_day)
            ctgan_data = self.gan_model.sample(100, randomize_samples=True)
            ctgan_data = make_raw(ctgan_data, i)
            ctgan_data.to_csv(os.path.join(
                Settings.generated_path, f'generate_day{i}.csv'), index=False)

            print(f"========End Day{i}=========\n\n")
        return None

    def growth_env(self, mode):
        """0 ~ 28일의 최대 생육환경 조성

        Args:
            mode (str): "0" => 0일의 생육환경
                        "other" => 1 ~ 28일의 생육환경
        """        
        folder_list = ["./max_data", "./generate_data"]

        for folder in folder_list:
            if os.path.isdir(folder) == False:
                os.makedirs(folder)

        if mode == "0":
            gen_data = pd.read_csv('./generate_data/generate_day0.csv')

            max_list = []
            for i in range(100):
                gen_data_new = gen_data[i * 24:(i + 1) * 24]
                gan_data = preprocess_day0(gen_data_new)
                y_pred = self.predict_model.predict(gan_data)
                max_list.append(y_pred.max())
                
            max_list = pd.DataFrame(max_list)
            idx_max = max_list.idxmax()
            print(f'idx_max: {idx_max}번째 index')
            
            gen_dat = gen_data[int(idx_max.values) * 24:(int(idx_max.values) + 1) * 24]
            gen_dat['DAT'] = 0
            gen_dat.to_csv("./max_data/day_0max.csv", index=False)

        if mode == "other":
            
            for day in range(1, 28):
                max_0 = pd.read_csv('./max_data/day_0max.csv')
                print(f'{day}일차 Start')
                
                gen_data = pd.read_csv(f"./generate_data/generate_day{day}.csv")  # 동적변경
                
                max_list = []
                for i in range(100):
                    gen_data_new = gen_data[i * 24:(i + 1) * 24]
                    gen_data_new = gen_data_new.reset_index(drop=True)
                    max_0['Case'] = i
                    gen_data_new['Case'] = i
                    gen_data_new['DAT'] = day
                    gan_data = preprocess_day_other(gen_data_new, max_0, day)
                    y_pred = self.predict_model.predict(gan_data)
                    max_list.append(y_pred.max())
                    
                max_list = pd.DataFrame(max_list)
                idx_max = max_list.idxmax()
                print('idx_max: %d번째 index' % idx_max)
                
                gen_dat = gen_data[int(idx_max.values) * 24:(int(idx_max.values) + 1) * 24]
                gen_dat['DAT'] = day
                con_dat = pd.concat([max_0, gen_dat], axis=0)
                con_dat.to_csv("./max_data/day_0max.csv", index=False)
                print(f'{day}일차 End')


if __name__ == '__main__':
    cls = CreateGE()
