using MultiClassification.IssueCategorizer;

Console.WriteLine("Multi Classification");

var testDataFilePath = Path.Combine(Environment.CurrentDirectory, "Data", "issues_test.tsv");
var trainDataFilePath = Path.Combine(Environment.CurrentDirectory, "Data", "issues_train.tsv");

var trainer = new Trainer();
trainer.Train(testDataFilePath);
Console.WriteLine(trainer.Evaluate(trainDataFilePath));
Console.WriteLine(trainer.Predict(
    title: "WebSockets communication is slow in my machine",
    description: "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."));

Console.ReadLine();