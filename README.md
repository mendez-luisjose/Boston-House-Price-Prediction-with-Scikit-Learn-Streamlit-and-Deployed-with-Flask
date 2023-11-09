# Boston House Price Prediction with Scikit Learn, Streamlit and Deployed with Flask

![](./preview/preview-1.gif)

## Boston House Price Prediction with Scikit Learn, Numpy, Pandas, Streamlit and Deployed with Flask 

The Model was trained with Tabular Boston House Prices Dataset and with the `XGBRegressor` Architecture. The Model predicts the Price of a given Boston House, also the U.I. to select the parameters of the House was built with Streamlit and the API with Flask. 

## Run it Locally

Test it Locally by running the `app.py` file, built with `Streamlit`, and the `api.py` file with `Flask`. Remember first to run the `api.py` file, copy the http url and saved in the API variable of the `app.py` file, and uncomment the code lines.

## App made with Streamlit
```sh
streamlit run app.py
```

## Deployed with Flash
```sh
python3 api.py
```

![](./preview/preview-2.gif)

## Resources
Boston House Prices Dataset: https://www.kaggle.com/datasets/arunjathari/bostonhousepricedata
