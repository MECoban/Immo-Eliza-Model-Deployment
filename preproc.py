import click
import joblib
import pandas as pd

@click.command()
@click.option("-i", "--input-dataset", help="path to input .csv dataset", required=True)
@click.option(
    "-o",
    "--output-dataset",
    default="output/predictions.csv",
    help="full path where to store predictions",
    required=True,
)
@click.option("-m", "--model-name", help="name of the model : LR for linear regression and LA for Lasso", required=True)
def predict(input_dataset, output_dataset, model_name):
    """Predicts house prices from 'input_dataset', stores it to 'output_dataset'."""
    ### -------- DO NOT TOUCH THE FOLLOWING LINES -------- ###
    # Load the data
    data = pd.read_csv(input_dataset)
    ### -------------------------------------------------- ###

    # Load the model artifacts using joblib
    #model_name = 'LinearRegression'

    if model_name == 'LA':
        model_name = 'Lasso'
    elif model_name == 'LR':
        model_name = 'LinearRegression'
    else:
        print(f"The model {model_name} has not been implemented yet !")

    artifacts = joblib.load(f"models/{model_name}.joblib")

    # Unpack the artifacts
    num_features = artifacts["features"]["num_features"]
    fl_features = artifacts["features"]["fl_features"]
    cat_features = artifacts["features"]["cat_features"]
    imputer = artifacts["imputer"]
    enc = artifacts["enc"]
    model = artifacts["model"]

    # Extract the used data
    data = data[num_features + fl_features + cat_features]


    # Removing the outliers
    mult = 20
    for feat in num_features:
        Q1 = data[feat].quantile(0.10)
        Q3 = data[feat].quantile(0.90)
        IQR = Q3 - Q1
        outliers = (data[feat] < (Q1-mult*IQR)) | (data[feat] > (Q3+mult*IQR))
        data[feat] = data[feat][~outliers]


    # Apply imputer and encoder on data
    data[num_features] = imputer.transform(data[num_features])
    data_cat = enc.transform(data[cat_features]).toarray()

    # Combine the numerical and one-hot encoded categorical columns
    data = pd.concat(
        [
            data[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(data_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    # Drop the NA
    data = data.dropna()

    # Make predictions
    predictions = model.predict(data)
    #predictions = predictions[:10]  # just picking 10 to display sample output :-)

    ### -------- DO NOT TOUCH THE FOLLOWING LINES -------- ###
    # Save the predictions to a CSV file (in order of data input!)
    pd.DataFrame({"predictions": predictions}).to_csv(output_dataset, index=False)

    # Print success messages
    click.echo(click.style("Predictions generated successfully!", fg="green"))
    click.echo(f"Saved to {output_dataset}")
    click.echo(
        f"Nbr. observations: {data.shape[0]} | Nbr. predictions: {predictions.shape[0]}"
    )
    ### -------------------------------------------------- ###


if __name__ == "__main__":
    # how to run on command line:
    # python .\predict.py -i "data\input.csv" -o "output\predictions.csv"
    predict()
