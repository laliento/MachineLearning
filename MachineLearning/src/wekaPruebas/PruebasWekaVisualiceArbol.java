package wekaPruebas;
import java.awt.BorderLayout;
import java.io.BufferedReader;
import java.io.FileReader;

import weka.associations.Apriori;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Remove;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

public class PruebasWekaVisualiceArbol {

	public static void main(String[] argss) throws Exception {
		String ruta= "C:\\Users\\vn0e623\\Documents\\Eduardo\\Ejemplos\\weka\\weather.arff";
		J48 j48Clase = new J48(); 
		Instances data = new Instances(new BufferedReader(new FileReader(ruta)));
		data.setClassIndex(data.numAttributes() - 1);
	     //uso de flitros
		 Discretize discretize = new Discretize();
		 discretize.setInputFormat(data);
		//remueve atribito/columna
		 Remove remove = new Remove();
		 remove.setOptions(weka.core.Utils.splitOptions("-R 1"));
		 remove.setInputFormat(data);
		 //discretize no supervizado, combierte un rango de numeros en los grupos señalados
		 weka.filters.unsupervised.attribute.Discretize discretizeNo = new weka.filters.unsupervised.attribute.Discretize();
		 discretizeNo.setInputFormat(data);
		 discretizeNo.setAttributeIndices("3");
		 discretizeNo.setBins(4);
		 data = Filter.useFilter(data,discretize);   // apply filter
		 //sin filtro
	     //data = new Instances(new BufferedReader(new FileReader("C:\\Users\\vn0e623\\Documents\\Eduardo\\Ejemplos\\weka\\weather.arff")));
	     j48Clase.buildClassifier(data);
	     //opciones de clasificador
	     j48Clase.setOptions(weka.core.Utils.splitOptions("-C 0.25 -M 2"));
	     //muestra texto
	     System.out.println("===================================Árbol de decisión (Valores Discretos)===================================");
	     System.out.println("el objetivo de los árboles de decisión es obtener reglas o relaciones que permitan clasificar a partir de los atributos.");
	     String[] options = new String[2];
		 options[0] = "-t";
		 options[1] = ruta;
		 System.out.println(Evaluation.evaluateModel(j48Clase, options));
		 //reglas de asociación
//		 para reglas de asociación apriori es necesario el filtro de Discretize!
//		 este filtro crea grupos (bins) para pasar de datos numéricos a clases
		 System.out.println("===================================Asociación  ( Apriori )===================================");
		 System.out.println("posibles relaciones o correlaciones entre distintas acciones o sucesos aparentemente independientes");
		  Apriori apriori = new Apriori();
		  apriori.setNumRules(10);
		  apriori.setClassIndex(data.classIndex() -1);
		  apriori.buildAssociations(data);
		  System.out.println(apriori.toString());
		  //cluster
//		  para poder generar el cluster kMeans se necesita de los datos originales (?)
		  System.out.println("===================================Clustering (k-medias)===================================");
		  System.out.println("trata de ordenar los ejemplos en una jerarquía según las regularidades en la distribución de los pares atributo-valor sin la guía del atributo especial clase");
	        SimpleKMeans kMeans = new SimpleKMeans();
	        kMeans.setSeed(10);
	        kMeans.setPreserveInstancesOrder(true);
	        kMeans.setNumClusters(3);
	        kMeans.buildClusterer(new Instances(new BufferedReader(new FileReader(ruta))));
	        int[] assignments = kMeans.getAssignments();
	        int i = 0;
	        for (int clusterNum : assignments) {
	            System.out.printf("Instance %d -> Cluster %d \n", i, clusterNum);
	            i++;
	        }
		  System.out.println(kMeans);
	     // muestra arbol
	     final javax.swing.JFrame jf = 
	       new javax.swing.JFrame("Weka Classifier Tree Visualizer Sentinel: J48");
	     jf.setSize(500,400);
	     jf.getContentPane().setLayout(new BorderLayout());
	     TreeVisualizer tv = new TreeVisualizer(null,
	    		 j48Clase.graph(),
	         new PlaceNode2());
	     jf.getContentPane().add(tv, BorderLayout.CENTER);
	     jf.addWindowListener(new java.awt.event.WindowAdapter() {
	       public void windowClosing(java.awt.event.WindowEvent e) {
	         jf.dispose();
	       }
	     });
	 
	     jf.setVisible(true);
	     tv.fitToScreen();  
	}
}
