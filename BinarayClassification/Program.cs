using BinarayClassification.SentimentAnalysis;

Console.WriteLine("Binary Classification");

var dataFilePath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");
var trainer = new Trainer();
trainer.Train(dataFilePath);
Console.WriteLine(trainer.Evaluate());

Console.WriteLine(trainer.Predict("I love smoked salmons"));

Console.ReadKey();
