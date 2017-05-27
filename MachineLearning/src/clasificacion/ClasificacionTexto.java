package clasificacion;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;

public class ClasificacionTexto {

	public static void main(String[] args) throws Exception {
		String[] options = new String[2];
		 options[0] = "-t";
		 options[1] = "C:\\Users\\vn0e623\\Documents\\Eduardo\\Ejemplos\\weka\\weather.arff";
		 System.out.println(Evaluation.evaluateModel(new J48(), options));
	}

}
