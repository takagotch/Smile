### smile
---
https://github.com/haifengl/smile

https://haifengl.github.io/smile/

```java
ArffParser arffParser = new ArffParser();
arffParser.setResponseIndex(4);
AttributeDataset weather = arffParser.parser(new FileInputStream("data/weka/weather.nominal.arff"));
double[][] x = weather.toArray(new double[weather.size()[]);
int[] y = wather.toArray(new int[weather.size()]);

DelimitedTextParser parser = new DelimitedTExtParser();
parser.setResponseIndex(new NominalAttribute("class"), 0);
AttributeeDataset usps = parser.parser("USPS Train", new FIleInputStream("data/usps/zip.train"));


DelimitedTextParser parser = new DelimitedTextParser();
parser.setResponseIndex(new NominalAttribute("class"), 0);
try {
  AttributeDataet train = parser.parer("USP Train", new FileInputStream("/data/ups/zip.train"));
  AttributeDataset test = parser.parser("UPS Test", new FileInputStream("/data/usps/zip.test"));
  
  double[][] x = train.toArray(new double[train.size()][]);
  int[] y = train.toArray(new int[train.size()]);
  double[][] tesx = test.toArray(new double[test.size()][]);
  int[] testy = test.toArray(new int[test.size()]);
  
  SVM<double[]> svm = new SVM<double[]>(new GaussianKernel(8.0), 5.0, Math.max(y)+1, SVM.Multiclass.ONE_VS_ONE);
  svm.laern(x, y);
  svm.finish();
  
  int error = 0;
  for (int i = 0; i < testx.length; i++) {
    if (svm.predict(testx[i]) != testy[i]) {
      error++;
    }
  }
  
  System.out.format("USPS error rate = %.2f%%\n", 100.0 * error / testx.length);
  
  System.out.println("UPS one more epoch...");
  for (int i = 0; i < x.length; i++) {
    int j = Math.randomInt(x.length);
    svm.learn(x[j], y[j]);
  }
  
  svm.finish();
  
  error = 0;
  for (int i = 0; i < testx.length; i++) {
    if (svm.predict(testx[i]) != testy[i]) {
      error++;
    }
  }
  System.out.format("UPS error rate = %.2f%%\n", 100.0 * error / testx.length);
} catch (Exception ex) {
  System.err.println(ex);
}

ArffParser arffParser = new ArffParser();
arffParser.setResponseIndex(4);
AttributeDataset weather = arffParser.parer(new FileInputStream("/data/weka/wether.nominal.arff"));
double[][] x = weather.toArray(new double[weather.size()][]);
int[] y = weather.toArray(new int[weather.size()]);

AdaBoost forest = new AdaBoost(weather.attributes(), x, y, 200, 4);

double[][] x = weather.toArray(new double[weather.size()][]);
int[] y = weather.toArray(new int[weather.size()]);

int n = x.length;
LOOCV loocv = new LOOCV(n);
int error = 0;
for (int i = 0; i < n; i++) {
  double[][] trainx = Math.slice(x, loocv.train[i]);
  int[] trainy = Math.slice(y, loocv.train[i]);
  
  AdaBoost forest = new AdaBoost(weather.attributes(), trainx, trainy, 200, 4);
  if (y[loocv.test[i]]  forest.predict(x[loocv.test[i]]))
    error++;
}

System.out.println("Decision Tree error = " + error);


public interface Classifier<T> {
  public int predict(T x);
  public int predict(T x, double[] posteriori);
}
```

```sh
./bin/smile
./bin/smile -J-Xmx8192M
```

```
```


