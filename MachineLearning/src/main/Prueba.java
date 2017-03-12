package main;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.Bagging;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.PART;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.M5P;
import weka.core.FastVector;
import weka.core.Instances;
 
public class Prueba {
	
	public static void main(String[] args) throws Exception {
		BufferedReader datafile = readDataFile(""
				+ "C:\\Users\\vn0e623\\Documents\\Eduardo\\Ejemplos\\weka\\weather.txt");
 
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
 
		// Do 10-split cross validation
		Instances[][] split = crossValidationSplit(data, 10);
 
		// Separate split into training and testing arrays
		Instances[] trainingSplits = split[0];
		Instances[] testingSplits = split[1];
 
		// Use a set of classifiers
		J48 j48 = new J48();
		j48.setOptions(weka.core.Utils.splitOptions("-C 0.25 -M 2"));
	
		 
		 String[] options = new String[2];
		 options[0] = "-t";
		 options[1] = "C:\\Users\\vn0e623\\Documents\\Eduardo\\Ejemplos\\weka\\weather.arff";
		 System.out.println(Evaluation.evaluateModel(new J48(), options));
		 System.out.println("xxxxxxxx");
//		j48.setOptions(options);
//		confidence factor, el cual controla el tamaño del árbol de decisión construido
//		No podemos controlar directamente el número de nodos,
//		pero cuanto más pequeño es este parámetro, mas simple tiende a ser el
//		clasificador (menos nodos), y viceversa. Este parámetro varía entre 0 y 1,
//		siendo el valor por omisión de 0.25
//		Vamos a ver si podemos construir clasificadores más simples y más complejos,
//		variando el confidence factor. Veamos que ocurre si ponemos 0.001 (árboles
//		simples) y 1.0 (árboles complejos) en ese parámetro.
//		j48.setConfidenceFactor(0.001f);
//		al procesar con Porcentage split
		Classifier[] models = { 
				j48 // Arboles de decisión
//				,new PART(), 
//				new DecisionTable(),//decision table majority classifier
//				new DecisionStump() //one-level decision tree
//				,new JRip(),//Aprendizaje de reglas: Ripper
//				new NaiveBayes(),//arbol
//				SMO
//				1) Transforma los datos a un espacio de dimensión superior.
//				2) Clasifica los datos mediante un hiperplano en esa dimensión.
//				Entrenamiento muy costoso. Muy buenos resultados.
//				http://www.youtube.com/watch?v=3liCbRZPrZA
//				new SMO()//Support Vector Machines
//				,new Bagging(),
//				new ZeroR()//primer algoritmo a probar y luego se usan los demas para poder igualarlo o superarlo
//				OneR
//				Este es uno de los clasificadores más sencillos y rápidos, aunque en ocasiones
//				sus resultados son sorprendentemente buenos en comparación con algoritmos
//				mucho más complejos. Simplemente selecciona el atributo que mejor “explica”
//				la clase de salida. Si hay atributos numéricos, busca los umbrales para hacer
//				reglas con mejor tasa de aciertos. OneR
//				,new OneR()
//				Regresion lineal
//				,new M5P()
		};
 
		// Run for each model
		for (int j = 0; j < models.length; j++) {
 
			// Collect every group of predictions for current model in a FastVector
			FastVector predictions = new FastVector();
 
			// For each training-testing split pair, train and test the classifier
			for (int i = 0; i < trainingSplits.length; i++) {
				Evaluation validation = classify(models[j], trainingSplits[i], testingSplits[i]);
 
				predictions.appendElements(validation.predictions());
 
				// Uncomment to see the summary for each training-testing pair.
				System.out.println(models[j].toString());
			}
 
			// Calculate overall accuracy of current classifier on all splits
			double accuracy = calculateAccuracy(predictions);
 
			// Print current classifier's name and accuracy in a complicated,
			// but nice-looking way.
			System.out.println("Accuracy of " + models[j].getClass().getSimpleName() + ": "
					+ String.format("%.2f%%", accuracy)
					+ "\n---------------------------------");
		}
 
	}
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;
 
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
 
		return inputReader;
	}
 
	public static Evaluation classify(Classifier model,
			Instances trainingSet, Instances testingSet) throws Exception {
		Evaluation evaluation = new Evaluation(trainingSet);
 
		model.buildClassifier(trainingSet);
		evaluation.evaluateModel(model, testingSet);
 
		return evaluation;
	}
 
	public static double calculateAccuracy(FastVector predictions) {
		double correct = 0;
 
		for (int i = 0; i < predictions.size(); i++) {
			NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
			if (np.predicted() == np.actual()) {
				correct++;
			}
		}
 
		return 100 * correct / predictions.size();
	}
 
	public static Instances[][] crossValidationSplit(Instances data, int numberOfFolds) {
		Instances[][] split = new Instances[2][numberOfFolds];
 
		for (int i = 0; i < numberOfFolds; i++) {
			split[0][i] = data.trainCV(numberOfFolds, i);
			split[1][i] = data.testCV(numberOfFolds, i);
		}
 
		return split;
	}
}
