import pandas as pd
from mwarehouse.forecasting.preprocessing import simple_ts_split, NnTsGenerator, TraditionalTsGenerator
from mwarehouse.forecasting.models.traditional import HoltWinters, Sarimax
from mwarehouse.forecasting.models.networks import multi_input, single_input
from mwarehouse.forecasting.postprocessing import Visualization
from mwarehouse.forecasting.preprocessing import DataFormater
import os 
import numpy as np

import seaborn as snscond
import matplotlib.pyplot as plt


def trainingStep(data,TimeColumn,Time_format,LabelColumn,Holdout,Resample_frequency,Epoches,Horizon,InputWidth,Shift,SizeTraining,SinCos,select_model_automatically,FillFuture):
    # --------------------------------------------- Argument Parsing --------------------------------------------------------

    print("--------------- Loading Arguments ----------------")

    # --------------------------------------------- Training Process ------------------------------------------------------
    print("--------------- Starting Preprocessing ----------------")
    formater = DataFormater(data, TimeColumn, LabelColumn, format = Time_format)

    #format date column and set as index
    formater.date_to_index()

    #convert non numerical data to either labels or one hot encoding
    formater.convert_category()

    #fill missing values in data
    formater.fill_missing()

    #we can either specify each column 
    '''
    formater.aggregate(Resample_frequency, sum_list=[
        'sales',
        'is_open',
        'has_promo',
        'is_state_holiday',
        'is_school_holiday',
        'future'], mean_list=[
        'year',
        'calender_week',
        'christmas',
        'easter',
        'public_holiday',
        'is_state_holiday_bool',
        'was_open_last_day',
        'last_promo_since_days',
        'next_promo_in_days',
        'state_holidays_in_last_7_days',
        'state_holidays_in_next_7_days',
        'school_holidays_in_last_7_days',
        'school_holidays_in_next_7_days',
    ], max_list=[
        'state_holidays_in_last_7_days',
        'state_holidays_in_next_7_days',
        'school_holidays_in_last_7_days',
        'school_holidays_in_next_7_days',
        'last_state_holiday_since_days',
        'last_school_holiday_since_days',
        'next_state_holiday_in_days',
        'next_school_holiday_in_days',
    ])
    '''
    #or just run mean over every column
    #formater.aggregate(frequency=Resample_frequency, all_mean = True)

    #normalize data and save the scaler to be able to invert the scaling later
    scaler = formater.normalize()[0]

    # reset index because next step expects a date column
    data = formater.get_df().reset_index()
    
    # Fill future frame for future predictions: In development! --> Only supported for data without Exogenous features!
    if FillFuture:
        print('Attention: Handle <Fill Future> with caution. Only supported for Data without Exogenous features. ! In Preview !')
        print('Attention: Currently not working as expected!')
        
        if data.shape[1] == 2:
            start = data[TimeColumn].max() + np.timedelta64(1,Resample_frequency)
            start = start.strftime(Time_format)
            
            # Create new (empty) timeFrame of time values
            time = pd.date_range(start, periods=Holdout + Shift, freq=Resample_frequency).to_numpy()
            
            # Empty Numpy Array
            empty = np.empty((time.shape[0],2))
            empty[:] = np.NaN
            
            # New Dataframe with empty array
            appending_df = pd.DataFrame(columns=[TimeColumn,LabelColumn],data=empty)
            appending_df[TimeColumn] = time
            
            # Combine Dataframes
            data = data.append(appending_df).reset_index(drop=True)
            data = data.fillna(method='ffill')
        else:
            raise TypeError("Error: Fill Future Frame only supports Data without exogenous features.")
    print(data.tail(30))

    print("--------------- Splitting Data ----------------")

    # --------------------------------------------- Data Wrangling ----------------------------------------------------
    sgl_train_df, sgl_val_df, sgl_test_df, sgl_future_frame = simple_ts_split(
        data,
        time_column=TimeColumn,
        holdout=Holdout,
        shift=Shift,
        train_size=SizeTraining,
        frequency=Resample_frequency,
        sin_cos_features=SinCos
    )

    multi_train_df, multi_val_df, multi_test_df, multi_future_frame = simple_ts_split(
        data,
        time_column=TimeColumn,
        holdout=Holdout + InputWidth - 1,
        shift=Shift,
        train_size=SizeTraining,
        frequency=Resample_frequency,
        sin_cos_features=SinCos
    )

    print("--------------- Creating Generator Objects ----------------")
    
    single_window = NnTsGenerator(
        train_df=sgl_train_df,
        val_df=sgl_val_df,
        test_df=sgl_test_df,
        input_width=1,
        label_width=1,
        shift=Shift,
        label_columns=[LabelColumn],
    )

    multi_window = NnTsGenerator(
        train_df=multi_train_df,
        val_df=multi_val_df,
        test_df=multi_test_df,
        input_width=InputWidth,
        label_width=1,
        shift=Shift,
        label_columns=[LabelColumn],
    )

    print("--------------- Starting Training ----------------")
        
    # NN Models
    single_dense_model = single_input.SingleDense(
        data_generator=single_window,
        future_frame=sgl_future_frame,
        scaler=scaler,
        max_epochs=Epoches
    )

    single_recurrent_model = single_input.SingleRecurrent(
        data_generator=single_window,
        future_frame=sgl_future_frame,
        scaler=scaler,
        max_epochs=Epoches
    )

    single_residual_model = single_input.SingleResidual(
        data_generator=single_window,
        future_frame=sgl_future_frame,
        scaler=scaler,
        max_epochs=Epoches
    )

    multi_dense_model = multi_input.MultiDense(
        data_generator=multi_window,
        future_frame=multi_future_frame,
        scaler=scaler,
        max_epochs=Epoches
    )

    multi_conv_model = multi_input.MultiConvolution(
        data_generator=multi_window,
        future_frame=multi_future_frame,
        scaler=scaler,
        max_epochs=Epoches
    )

    multi_recurrent_model = multi_input.MultiRecurrent(
        data_generator=multi_window,
        future_frame=multi_future_frame,
        scaler=scaler,
        max_epochs=Epoches
    )

    print("--------------- Selecting Model for Baseline Model ----------------")
    selected_model = single_dense_model
    selected_model_name = 'single_dense_model'

    if select_model_automatically:
        print('--------------- Initial Model Selection ---------------')
        # NN Models
        single_dense_model.fit(
            baseline=True
        )
        single_recurrent_model.fit(
            baseline=True
        )
        single_residual_model.fit(
            baseline=True
        )
        multi_dense_model.fit(
            baseline=True
        )
        multi_conv_model.fit(
            baseline=True
        )
        multi_recurrent_model.fit(
            baseline=True
        )

        print('MAPES of during Model Selection: ')
        print('Single Dense Model:', single_dense_model.initial_mape)
        print('Single Recurrent Model:', single_recurrent_model.initial_mape)
        print('Single Residual Model:', single_residual_model.initial_mape)
        print('Multi Dense Model:', multi_dense_model.initial_mape)
        print('Multi Conv Model:', multi_conv_model.initial_mape)
        print('Multi Recurrent Model:', multi_recurrent_model.initial_mape)

        print('\n\nInitial Model selection in terms of MAPE: ')
        model_accs = list()

        model_accs.append(single_dense_model.initial_mape)
        model_accs.append(single_recurrent_model.initial_mape)
        model_accs.append(single_residual_model.initial_mape)
        model_accs.append(multi_dense_model.initial_mape)
        model_accs.append(multi_conv_model.initial_mape)
        model_accs.append(multi_recurrent_model.initial_mape)

        best = model_accs.index(min(model_accs))
        if best == 0:
            print('BEST MODEL IS:', 'single_dense_model')
            selected_model = single_dense_model
            selected_model_name = 'single_dense_model'
        elif best == 1:
            print('BEST MODEL IS:', 'single_recurrent_model')
            selected_model = single_recurrent_model
            selected_model_name = 'single_recurrent_model'
        elif best == 2:
            print('BEST MODEL IS:', 'single_residual_model')
            selected_model = single_residual_model
            selected_model_name = 'single_residual_model'
        elif best == 3:
            print('BEST MODEL IS:', 'multi_dense_model')
            selected_model = multi_dense_model
            selected_model_name = 'multi_dense_model'
        elif best == 4:
            print('BEST MODEL IS:', 'multi_conv_model')
            selected_model = multi_conv_model
            selected_model_name = 'multi_conv_model'
        elif best == 5:
            print('BEST MODEL IS:', 'multi_recurrent_model')
            selected_model = multi_recurrent_model
            selected_model_name = 'multi_recurrent_model'
        

    print("--------------- Retrain Baseline Model ----------------")
    # Baseline
    selected_model.fit(baseline=True)
    
    
    print("--------------- Selecting Model for HP Search ----------------")
    # Hyperparameter Search
    selected_model.fit(hp_search = True)

    print("--------------- Fitting final Model ----------------")
    # Final Training
    selected_model.fit(final_training=True)

    print("--------------- Get Predictions----------------")
    raw_prediction_frame, predictions = selected_model.final_predictions(horizon=Horizon, frequency=Resample_frequency,
                                                                        time_column=TimeColumn)
    
    print("--------------- Format Predictions----------------")
    #unscale data, date needs to be removed and added back for this process
    data.set_index(TimeColumn, inplace = True)
    data = pd.DataFrame(data=scaler.inverse_transform(data), columns=data.columns, index = data.index)
    data.reset_index(inplace=True)

    print("--------------- Visualization currently not available----------------")
    vis = Visualization(data, TimeColumn, LabelColumn, predictions)
    
    print(vis.prediction)
    
    
    return vis, selected_model, selected_model_name