import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from calendar import day_abbr, month_abbr, mdays


def create_dataframe(dataframe, cls):

    # define the names of the columns for the DataFrame
    dataframe.columns = ['year', 'month', 'day', 'hour', 'global radiation', 'diffuse radiation',
                         'temp', 'wind speed','relative humidity', 'cloud cover', 'precipitation']

    # create a new DataFrame 'datetime' containing all columns between 'year' and 'hour'
    datetime = dataframe.loc[:, 'year':'hour']

    # adjust the 'hour' column values from 1 --> 24 to 0 --> 23 format
    datetime['hour'] = datetime['hour'] - 1

    # dreate a new 'DateTime' column by combining 'year', 'month', 'day', and 'hour'
    datetime['DateTime'] = datetime.apply(lambda row: dt.datetime(row.year, row.month, row.day, row.hour), axis=1)

    # Convert the 'DateTime' column to a pandas datetime format
    datetime['DateTime'] = pd.to_datetime(datetime.DateTime)

    # Set the DataFrame's index to be a DatetimeIndex based on the 'DateTime' column
    dataframe.index = pd.DatetimeIndex(datetime.DateTime)

    # Include the 'class' column for each city with values specified by 'cls'
    dataframe['class'] = np.full(shape=dataframe.shape[0], fill_value=cls)

    # Drop the first four columns ('year', 'month', 'day', 'hour') as they are no longer needed
    dataframe = dataframe.drop(['year', 'month', 'day', 'hour'], axis=1)

    # Return the modified DataFrame
    return dataframe



def plot_3d_vis(Y, y, label_1, label_2, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(Y[:, 0][y == 1], Y[:, 1][y == 1], Y[:, 2][y == 1], marker='x', s=50, label=label_1)
    ax.scatter(Y[:, 0][y == 2], Y[:, 1][y == 2], Y[:, 2][y == 2], marker='*', s=50, label=label_2)

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title(title)
    plt.legend()
    plt.show()


def extract_csv_from_excel():
    # Specify the Excel file path which contains the data downloaded from TfL website
    excel_file_path = './lab7-data/tfl-data/tfl-daily-cycle-hires.xlsx'

    # Read the 'Metadata' sheet from the Excel file
    metadata_df = pd.read_excel(excel_file_path, sheet_name='Metadata')

    # Read the 'Data' sheet from the Excel file
    data_df = pd.read_excel(excel_file_path, sheet_name='Data')

    # Save the 'Metadata' DataFrame as a CSV file
    metadata_df.to_csv('./lab7-data/tfl-data/metadata.csv', index=False)

    # Save the 'Data' DataFrame as a CSV file for later use
    data_df.to_csv('./lab7-data/tfl-data/tfl-daily-cycle-hires.csv', index=False)


def load_tfl_cycle_data():
    cycle_df = pd.read_csv('./lab7-data/tfl-data/tfl-daily-cycle-hires.csv').iloc[:, :2]
    cycle_df = cycle_df.rename(columns={'Day':'datetime'})
    cycle_df['datetime'] = pd.to_datetime(cycle_df['datetime'])
    cycle_df = cycle_df.set_index('datetime', drop=True)
    cycle_df = cycle_df.loc['2011-01-01':, :]

    return cycle_df


def plot_heatmap(df, title, x_label, y_label):
    # create a figure and axes for the heatmap
    f, ax = plt.subplots(figsize=(12,6))

    # create a heatmap specifying the data, axis, colormap, and colorbar
    sns.heatmap(df, ax = ax, cmap=plt.cm.viridis, cbar_kws={'boundaries':np.arange(10000,45000,5000)})

    # get the colorbar axis
    cbax = f.axes[1]

    # set the font size for colorbar tick labels
    [l.set_fontsize(13) for l in cbax.yaxis.get_ticklabels()]

    # set the colorabar label
    cbax.set_ylabel('Santander cycles hires', fontsize=13)

    # add horizontal gridlines
    [ax.axhline(x, ls=':', lw=0.5, color='0.8') for x in np.arange(1, 7)]

    # add vertical gridlines
    [ax.axvline(x, ls=':', lw=0.5, color='0.8') for x in np.arange(1, 24)]

    # add title to the heatmap
    ax.set_title(title, fontsize=16)

    # set font size for x-axis tick labels
    [l.set_fontsize(13) for l in ax.xaxis.get_ticklabels()]

    # set font size for y-axis tick labels
    [l.set_fontsize(13) for l in ax.yaxis.get_ticklabels()]

    # set x-axis label
    ax.set_xlabel(x_label, fontsize=15)

    # set y-axis label
    ax.set_ylabel(y_label, fontsize=15)

    if y_label == 'Day of the week':
        # use the 'day_abbr' list to label days of the week
        ax.set_yticklabels(day_abbr[0:7])
    elif y_label == 'Year':
        # rotate the ticklabels for the y-axis
        ax.set_yticklabels(np.arange(2011, 2021, 1), rotation=0);



def prepare_data(data, target_feature):
    """
    prepare the data for ingestion by fbprophet:
    see: https://facebook.github.io/prophet/docs/quick_start.html
    """
    # create copy of the data
    new_data = data.copy()

    # reset the index of the DataFrame to numeric integers
    new_data.reset_index(inplace=True)

    # rename the columns 'datetime' to 'ds' and the target feature (in this case called 'target_feature) to 'y'
    new_data = new_data.rename({'datetime':'ds', '{}'.format(target_feature):'y'}, axis=1)

    return new_data



def train_test_split(data):
    # set the 'ds' column as the index and extract rows up to July 31, 2020 for the training set
    train = data.set_index('ds').loc[:'2020-07-31', :].reset_index()

    # set the 'ds' column as the index and extract rows from August 1, 2020, for the testing set
    test = data.set_index('ds').loc['2020-08-01':, :].reset_index()

    # return the training and testing sets as seperate DataFrames
    return train, test



def make_predictions_df(forecast, data_train, data_test):
    """
    Function to convert the output Prophet dataframe to a datetime index and append the actual target values at the end
    """

    # convert the 'ds' column in the forecast DataFrame to a datetime index
    forecast.index = pd.to_datetime(forecast.ds)

    # convert the 'ds' column in the data_train DataFrame to a datetime index
    data_train.index = pd.to_datetime(data_train.ds)

    # convert the 'ds' column in the data_test DataFrame to a datetime index
    data_test.index = pd.to_datetime(data_test.ds)

    # concatenate the training and testing sets into a single DataFrame
    data = pd.concat([data_train, data_test], axis=0)

    # append the actual target values ('y') from the concatenated data DataFrame to the forecast DataFrame
    forecast.loc[:,'y'] = data.loc[:,'y']

    return forecast



def plot_predictions(forecast, start_date):
    """
    Function to plot the predictions
    """

    # create a figure and axis for the plot with a specific figsize
    f, ax = plt.subplots(figsize=(14, 8))

    # extract the data from the forecast corresponding to the training period (up to '2020-07-31')
    train = forecast.loc[start_date:'2020-07-31',:]

    # plot actual values as black markers
    ax.plot(train.index, train.y, 'ko', markersize=3)

    # plot predicted values as a blue line
    ax.plot(train.index, train.yhat, color='steelblue', lw=0.5)

    # fill the uncertainty interval with a light blue color
    ax.fill_between(train.index, train.yhat_lower, train.yhat_upper, color='steelblue', alpha=0.3)

    # extract the data from the forecast corresponding to the testing period (from '2020-08-01' onwards)
    test = forecast.loc['2020-08-01':,:]

    # plot actual values as red markers
    ax.plot(test.index, test.y, 'ro', markersize=3)

    # plot the predicted values as a coral line
    ax.plot(test.index, test.yhat, color='coral', lw=0.5)

    # fill the uncertainty interval with a light coral color
    ax.fill_between(test.index, test.yhat_lower, test.yhat_upper, color='coral', alpha=0.3)

    # add a vertical dashed line to mark the separation point between training and testing data
    ax.axvline(forecast.loc['2020-08-01', 'ds'], color='k', ls='--', alpha=0.7)

    # add gridlines
    ax.grid(ls=':', lw=0.5)

    return f, ax



def create_joint_plot(forecast, x='yhat', y='y', title=None):
    # create a joint plot with 'yhat' as the x-axis and 'y' as the y-axis
    g = sns.jointplot(x='yhat', y='y', data=forecast, kind="reg", color="b")

    # set the width and height of the figure
    g.fig.set_figwidth(8)
    g.fig.set_figheight(8)

    # access the second subplot in the figure (scatter plot) and set its title if provided
    ax = g.fig.axes[1]
    if title is not None:
        ax.set_title(title, fontsize=16)

    # access the first subplot in the figure (histograms) and display the correlation coefficient
    ax = g.fig.axes[0]
    ax.text(5000, 60000, "R = {:+4.2f}".format(forecast.loc[:,['y','yhat']].corr().iloc[0,1]), fontsize=16)

    # set labels, limits, and grid lines for the x and y axes
    ax.set_xlabel('Predictions', fontsize=15)
    ax.set_ylabel('Observations', fontsize=15)
    ax.set_xlim(0, 80000)
    ax.set_ylim(0, 80000)
    ax.grid(ls=':')

    # set the font size for the x-axis and y-axis tick labels
    [l.set_fontsize(13) for l in ax.xaxis.get_ticklabels()]
    [l.set_fontsize(13) for l in ax.yaxis.get_ticklabels()];

    # add gridlines
    ax.grid(ls=':')



def is_daytime_hour(datetime):
    # check if the hour component falls between 6 and 20 (inclusive)
    if 6 <= datetime.hour <= 20:
        return 1 # return 1 if it's a daytime hour
    else:
        return 0 # return 0 it it's not a daytime hour


def is_pandemic_affected(ds):
    # convert the input 'ds' to a pandas datetime object
    date = pd.to_datetime(ds)
    # check if the year component of the date is equal to 2020
    return date.year == 2020