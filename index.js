const tf = require("@tensorflow/tfjs")

// Construct training data
const trainingData = [
	{ input: [0, 1], output: [1, 0, 0] }, // Liverpool win
	{ input: [0, 1], output: [0, 0, 1] }, // Draw
	{ input: [0, 1], output: [1, 0, 0] }, // Liverpool win
	{ input: [0, 1], output: [1, 0, 0] }, // Liverpool win
	{ input: [0, 1], output: [0, 1, 0] }, // Liverpool lose
	{ input: [0, 1], output: [0, 0, 1] }, // Draw
	{ input: [0, 1], output: [0, 0, 1] }, // Draw
	{ input: [0, 1], output: [1, 0, 0] }, // Liverpool win
	{ input: [0, 1], output: [1, 0, 0] }, // Liverpool win
	{ input: [0, 1], output: [0, 0, 1] }, // Draw
	{ input: [0, 1], output: [1, 0, 0] }, // Liverpool win
	{ input: [0, 1], output: [0, 0, 1] }, // Draw
	{ input: [0, 1], output: [1, 0, 0] }, // Liverpool win
	{ input: [0, 1], output: [0, 0, 1] }, // Draw
	{ input: [0, 1], output: [1, 0, 0] }, // Liverpool win
	{ input: [0, 1], output: [1, 0, 0] }, // Liverpool win
	{ input: [0, 1], output: [1, 0, 0] }, // Liverpool win
	{ input: [0, 1], output: [0, 0, 1] }, // Draw
	{ input: [0, 1], output: [0, 0, 1] }, // Draw
	{ input: [0, 1], output: [0, 0, 1] }, // Draw
]

// Step 2: Construct the model
const model = tf.sequential()
// creates a model where each layer is connected
// to the next in a sequential manner
model.add(tf.layers.dense({ units: 10, inputShape: [2], activation: "relu" }))
// This is a hidden layer.
// Allows the model to learn more complex representations
// and extract meaningful features from the input data.
// inputShape: [2] indicates our 2 inputs: Liverpool and Everton.
// relu is used for introducing non-linearity in neural networks.

model.add(tf.layers.dense({ units: 3, activation: "softmax" }))
// This is the output layer.
// Generates probabilities for each outcome,
// indicating the likelihood of each outcome occurring.
// units: 3 represents the probability our 3 outputs (win, lose, draw).
// Softmax activation function is used for multi-class classification problems
// as it converts the output values into probabilities that sum up to 1.

// compile the model
model.compile({ loss: "categoricalCrossentropy", optimizer: "adam" })
// Categorical cross-entropy measures the dissimilarity between the predicted
// probabilities and the true labels, which makes the model output higher
// probabilities for the correct class.
// adam is an optimization algorithm that adapts the learning rate during
// training and combines the benefits of both AdaGrad and RMSProp optimizers.
// It is widely used for training neural networks and helps in efficiently
// updating the model's parameters to minimize the loss.

// Step 4: Prepare the training data
const xTrain = tf.tensor2d(trainingData.map((item) => item.input))
const yTrain = tf.tensor2d(trainingData.map((item) => item.output))

// Step 5: Train the model
async function trainModel() {
	await model.fit(xTrain, yTrain, { epochs: 100 })
	console.log("Training complete!")
}
// epochs: 100 - the model will iterate over the entire training dataset 100
// times and will update its internal parameters (weights and biases)
// in each iteration. This will minimise the loss and improve its performance.

// Step 6: Make predictions
function makePredictions() {
	const testData = [
		{ input: [0, 1] }, // Liverpool vs. Everton
	]
	const predictions = model.predict(
		tf.tensor2d(testData.map((item) => item.input))
	)
	const predictedResults = Array.from(predictions.dataSync())
	//  uses the trained model to make predictions on the test data

	testData.forEach((data, index) => {
		const predictedOutcome = predictedResults.slice(
			index * 3,
			(index + 1) * 3
		)
		console.log(
			`Match ${index + 1}: Liverpool ${data.input[0]} vs. Everton ${
				data.input[1]
			}`
		)
		// extracts the predicted outcome for our test case
		// Since each prediction consists of three values (win, lose, draw),
		// slice() is used to extract the relevant values for the current test case.
		console.log("Predicted outcome:", predictedOutcome)
	})
}

// Step 7: Train the model and make predictions
trainModel().then(() => {
	makePredictions()
})

