import java.io.{BufferedWriter, File, FileWriter, PrintWriter}

import breeze.linalg.{*, DenseMatrix, DenseVector, csvread}
import breeze.stats.mean
import com.stripe.rainier.compute._
import com.stripe.rainier.core.{Normal, RandomVariable, _}
import com.stripe.rainier.sampler._

import scala.collection.immutable.ListMap
import scala.annotation.tailrec

/**
 * Variable selection using the Horseshoe version 3. Uses Avi Bryand's suggestion for half-Cauchy using .abs
 */

/**
 * Builds a 2-way Anova model with variable selection in Scala using Rainier version 0.2.2
 * Main effects: a and b. Interaction effects: gamma. Apply variable selection on the interactions using the Horseshoe prior
 * Model: X_ijk | mu, a_j, b_k, gamma_jk, tau, lambda_jk, tauHS  ~ N(mu + a_j + b_k + gamma_jk , τ^−1 )
 */
object HorseshoeNew {

  val REAL_ONE = Real(1.0)
  val REAL_ZERO_POINT_FIVE = Real(0.5)

  def main(args: Array[String]): Unit = {
    val rng = ScalaRNG(3)
    val inputFilePath = "./SimulatedDataAndTrueCoefs/simulDataWithInters.csv"
    val outputFilePath = "./asymmetricBothRainierHorseshoe10x151mHMC300AllSaved.csv"
    val runtimeFilePath = "./asymmetricBoth10x15RainierHorseshoeHMC300Runtime1mAllSaved.txt"
    val (data, n1, n2) = dataProcessing(inputFilePath)
    val HMCResultsHS = mainEffectsAndInters(data, rng, n1, n2, runtimeFilePath)
    printAndSaveResults(HMCResultsHS, n1, n2, outputFilePath)
  }

  /**
   * Calculate execution time
   */
  def time[A](f: => A, runtimeFilePath: String): A = {
    val s = System.nanoTime
    val ret = f
    val execTime = (System.nanoTime - s) / 1e6
    println("time: " + execTime + "ms")
    val bw = new BufferedWriter(new FileWriter(new File("./asymmetricBoth10x15RainierHorseshoeHMC300Runtime1mAllSaved.txt")))
    bw.write(execTime.toString)
    bw.close()
    ret
  }

  /**
   * Process data read from input file
   */
  def dataProcessing(inputFilePath: String): (Map[(Int, Int), List[Double]], Int, Int) = {
    val data = csvread(new File(inputFilePath))
    val y = data(::, 0).toArray
    val alpha = data(::, 1).map(_.toInt)
    val beta = data(::, 2).map(_.toInt)
    val nj = alpha.toArray.distinct.length //the number of levels for the first variable
    val nk = beta.toArray.distinct.length //the number of levels for the second variable
    val l = alpha.length
    var dataList = List[(Int, Int)]()

    for (i <- 0 until l) {
      dataList = dataList :+ (alpha(i), beta(i))
    }

    // Create a Map[(Int, Int), List[Double]] to represent the data. Key: (levelForFirstVariable, levelForSecondVariable), value: List of observations
    val dataMap = (dataList zip y).groupBy(_._1).map { case (k, v) => ((k._1 - 1, k._2 - 1), v.map(_._2)) }
    (dataMap, nj, nk)
  }

  /**
   * Use Rainier for modelling main and interaction effects. Apply variable selection to the interactions using Horseshoe priors
   */
  def mainEffectsAndInters(dataMap: Map[(Int, Int), List[Double]], rngS: ScalaRNG, n1: Int, n2: Int, runtimeFilePath: String): scala.List[Map[String, Map[(Int, Int), Double]]] = {
    implicit val rng = rngS

    // Implementation of sqrt for Real
    def sqrtF(x: Real): Real = {
      //      (REAL_ZERO_POINT_FIVE * x.log).exp
      x.pow(REAL_ZERO_POINT_FIVE)
    }

    def updatePrior(mu: Real, sdE1: Real, sdE2: Real, sdG: Real, sdDR: Real): scala.collection.mutable.Map[String, Map[(Int, Int), Real]] = {
      val myMap = scala.collection.mutable.Map[String, Map[(Int, Int), Real]]()

      myMap("mu") = Map((0, 0) -> mu)
      myMap("eff1") = Map[(Int, Int), Real]()
      myMap("eff2") = Map[(Int, Int), Real]()
      myMap("effg") = Map[(Int, Int), Real]()
      myMap("lambdajk") = Map[(Int, Int), Real]()
      myMap("sigE1") = Map((0, 0) -> sdE1)
      myMap("sigE2") = Map((0, 0) -> sdE2)
      myMap("sdHS") = Map((0, 0) -> sdG)
      myMap("sigD") = Map((0, 0) -> sdDR)

      myMap
    }

    //Define the prior
    //Jags uses precision and we have: mu ~ dnorm(0, 0.0001), here we use sd = sqrt(1/tau)
    val prior = for {
      mu <- Normal(0, 100).param

      // Sample tau, estimate sd to be used in sampling from Normal the effects for the 1st variable
      tauE1RV = Gamma(1, 10000).param //RandomVariable[Real]
      tauE1 <- tauE1RV //Real
      sdE1 = sqrtF(REAL_ONE / tauE1) //Real. Without Real() it is Double

      // Sample tau, estimate sd to be used in sampling from Normal the effects for the 2nd variable
      tauE2RV = Gamma(1, 10000).param
      tauE2 <- tauE2RV
      sdE2 = sqrtF(REAL_ONE / tauE2)

      // Sample tHS for the interaction effects
      tHSRV = Cauchy(0,1).param
      tHS <- tHSRV
      sdHS = sqrtF(REAL_ONE / tHS.abs)

      // Sample tau, estimate sd to be used in sampling from Normal for fitting the model
      tauDRV = Gamma(1, 10000).param
      tauD <- tauDRV
      sdDR = sqrtF(REAL_ONE / tauD)
      //scala.collection.mutable.Map("mu" -> Map((0, 0) -> mu), "eff1" -> Map[(Int, Int), Real](), "eff2" -> Map[(Int, Int), Real](), "effg" -> Map[(Int, Int), Real](), "sigE1" -> Map((0, 0) -> sdE1), "sigE2" -> Map((0, 0) -> sdE2), "sigInter" -> Map((0, 0) -> sdG), "sigD" -> Map((0, 0) -> sdDR))
    } yield updatePrior(mu, sdE1, sdE2, sdHS, sdDR)

    /**
     * Helper function to update the values for the main effects of the Map
     */
    def updateMap(myMap: scala.collection.mutable.Map[String, Map[(Int, Int), Real]], index: Int, key: String, addedValue: Real): scala.collection.mutable.Map[String, Map[(Int, Int), Real]] = {
      myMap(key) += ((0, index) -> addedValue)
      myMap
    }

    /**
     * Add the main effects of alpha
     */
    def addAplha(current: RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]], i: Int): RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]] = {

      for {
        cur <- current
        gm_1 <- Normal(0, cur("sigE1")(0, 0)).param
      } yield updateMap(cur, i, "eff1", gm_1)
    }

    /**
     * Add the main effects of beta
     */
    def addBeta(current: RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]], j: Int): RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]] = {
      for {
        cur <- current
        gm_2 <- Normal(0, cur("sigE2")(0, 0)).param
      } yield updateMap(cur, j, "eff2", gm_2)
    }

    /**
     * Helper function to update the values for the interaction effects of the Map
     */
    def updateMapGammaEffLambda(myMap: scala.collection.mutable.Map[String, Map[(Int, Int), Real]], i: Int, j: Int, key1: String, addedValue1: Real, key2: String, addedValue2: Real): scala.collection.mutable.Map[String, Map[(Int, Int), Real]] = {
      myMap(key1) += ((i, j) -> addedValue1)
      myMap(key2) += ((i, j) -> addedValue2)
      myMap
    }

    /**
     * Add the interaction effects of beta
     */
    def addGammaEff(current: RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]], i: Int, j: Int): RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]] = {
      for {
        cur <- current
        lambdajk <- Cauchy(0,1).param.map(_.abs)
        gm_inter <- Normal(0, 1/(cur("sdHS")(0, 0) * lambdajk)).param
        //yield Map("mu" -> cur("mu"), "eff1" -> cur("eff1"), "eff2" -> (cur("eff2") += ((0, j) -> gm_2)), "sdE1" -> cur("sdE1"), "sdE2" -> cur("sdE2"), "sdD" -> cur("sdDR"))
      } yield updateMapGammaEffLambda(cur, i, j, "effg", gm_inter, "lambdajk", lambdajk)
    }

    /**
     * Add the interaction effects
     */
    def addGamma(current: RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]]): RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]] = {
      var tempAlphaBeta: RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]] = current

      for (i <- 0 until n1) {
        for (j <- 0 until n2) {
          tempAlphaBeta = addGammaEff(tempAlphaBeta, i, j)
        }
      }
      tempAlphaBeta
    }

    /**
     * Fit to the data per group
     */
    def addGroup(current: RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]], i: Int, j: Int, dataMap: Map[(Int, Int), List[Double]]): RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]] = {
      for {
        cur <- current
        gm = cur("mu")(0, 0) + cur("eff1")(0, i) + cur("eff2")(0, j) + cur("effg")(i, j)
        _ <- Normal(gm, cur("sigD")(0, 0)).fit(dataMap(i, j))
      } yield cur
    }

    /**
     * Add all the main effects for each group. Version: Recursion
     */
    @tailrec def addAllEffRecursive(alphabeta: RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]], dataMap: Map[(Int, Int), List[Double]], i: Int, j: Int): RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]] = {

      val temp = addGroup(alphabeta, i, j, dataMap)

      if (i == n1 - 1 && j == n2 - 1) {
        temp
      } else {
        val nextJ = if (j < n2 - 1) j + 1 else 0
        val nextI = if (j < n2 - 1) i else i + 1
        addAllEffRecursive(temp, dataMap, nextI, nextJ)
      }
    }

    /**
     * Add all the main effects for each group. Version: Loop
     */
    def addAllEffLoop(alphabeta: RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]], dataMap: Map[(Int, Int), List[Double]]): RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]] = {

      var tempAlphaBeta: RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]] = alphabeta

      for (i <- 0 until n1) {
        for (j <- 0 until n2) {
          tempAlphaBeta = addGroup(tempAlphaBeta, i, j, dataMap)
        }
      }
      tempAlphaBeta
    }

    /**
     * Add all the main effects for each group.
     * We would like something like: val result = (0 until n1)(0 until n2).foldLeft(alphabeta)(addGroup(_,(_,_)))
     * But we can't use foldLeft with double loop so we use an extra function either with loop or recursively
     */
    def fullModel(alphabeta: RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]], dataMap: Map[(Int, Int), List[Double]]): RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]] = {
      addAllEffRecursive(alphabeta, dataMap, 0, 0)
    }

    // Add the effects sequentially
    val alpha = (0 until n1).foldLeft(prior)(addAplha(_, _))
    val alphabeta = (0 until n2).foldLeft(alpha)(addBeta(_, _))
    val alphabetagamma = addGamma(alphabeta)
    val fullModelRes = fullModel(alphabetagamma, dataMap)

    val model = for {
      mod <- fullModelRes
    } yield Map("mu" -> mod("mu"),
      "eff1" -> mod("eff1"),
      "eff2" -> mod("eff2"),
      "effg" -> mod("effg"),
      "lambdajk" -> mod("lambdajk"),
      "sdHS" -> mod("sdHS"),
      "sigE1" -> mod("sigE1"),
      "sigE2" -> mod("sigE2"),
      "sigD" -> mod("sigD"))

    // Sampling
    println("Model built. Sampling now (will take a long time)...")
    val thin = 10
    val out = time(model.sample(HMC(300), 10000, 10000 * thin, thin), runtimeFilePath)
    println("Sampling finished.")
    out
  }

  /**
   * Process the result of sampling, print and save the results
   */
  def printAndSaveResults(out: scala.List[Map[String, Map[(Int, Int), Double]]], n1: Int, n2: Int, outputFilePath: String) = {

    /**
     * Select the results for a specific variable and convert them to DenseMatrix
     */
    def variableDM(varName: String):  DenseMatrix[Double] ={
      // Separate the data for the specific variable of interest
      val sepVariableData = out
        .flatMap{ eff1ListItem => eff1ListItem(varName) }
        .groupBy(_._1)
        .map { case (k, v) => k -> v.map(_._2) }

      // If the map contains more than one keys, they need to be sorted out to express the effects sequentially.
      // This is necessary to compare the results from Rainier, Scala and R in R
      val tempData= {
        varName match {
          case "mu" | "sigE1" | "sigE2" | "sigD" | "sdHS" => sepVariableData
          case "eff1" | "eff2" => ListMap(sepVariableData.toSeq.sortBy(_._1._2):_*)
          case "effg" | "lambdajk" => ListMap(sepVariableData.toSeq.sortBy(_._1._2).sortBy(_._1._1):_*)
        }
      }
      val tempList = tempData.map{case (k,listDb) => (listDb)}.toList
      DenseMatrix(tempList.map(_.toArray): _*).t
    }

    println("----------------mu ------------------")
    val muMat = variableDM("mu")
    println(mean(muMat(::,*)))

    println("----------------eff1 ------------------")
    val effects1Mat = variableDM("eff1")
    println(mean(effects1Mat(::,*)))

    println("----------------eff2 ------------------")
    val effects2Mat = variableDM("eff2")
    println(mean(effects2Mat(::,*)))

    println("----------------effg in the order: (1,1),(1,2), (1,3),...,(2,1) etc... ------------------")
    val effgMat = variableDM("effg")
    println(mean(effgMat(::,*)))

    println("----------------lambdajk in the order: (1,1),(1,2), (1,3),...,(2,1) etc... ------------------")
    val effLambdaMat = variableDM("lambdajk")
    println(mean(effLambdaMat(::,*)))

    println("----------------sdHS ------------------")
    val sigHSMat = variableDM("sdHS")
    println(mean(sigHSMat(::,*)))

    println("----------------sigΕ1 ------------------")
    val sigE1Mat = variableDM("sigE1")
    println(mean(sigE1Mat(::,*)))

    println("----------------sigΕ2 ------------------")
    val sigE2Mat = variableDM("sigE2")
    println(mean(sigE2Mat(::,*)))

    println("----------------sigD ------------------")
    val sigDMat = variableDM("sigD")
    println(mean(sigDMat(::,*)))

    val results = DenseMatrix.horzcat(muMat, sigDMat, sigE1Mat, sigE2Mat, sigHSMat, effects1Mat, effects2Mat, effgMat, effLambdaMat )

    val outputFile = new File(outputFilePath)

    printTitlesToFile(results, n1, n2, outputFile )

    /**
     * Save results to csv files
     */
    def printTitlesToFile(resMat: DenseMatrix[Double], n1: Int, n2: Int, outputFile: File): Unit = {
      val pw = new PrintWriter(outputFile)

      val gammaTitles = (1 to n1)
        .map { i => "-".concat(i.toString) }
        .map { entry =>
          (1 to n2).map { j => "gamma".concat(j.toString).concat(entry) }.mkString(",")
        }.mkString(",")

      val lambdaTitles = (1 to n1)
        .map { i => "-".concat(i.toString) }
        .map { entry =>
          (1 to n2).map { j => "lambda".concat(j.toString).concat(entry) }.mkString(",")
        }.mkString(",")

      pw.append("mu ,sigD, sigE1, sigE2, sigHS,")
        .append( (1 to n1).map { i => "alpha".concat(i.toString) }.mkString(",") )
        .append(",")
        .append( (1 to n2).map { i => "beta".concat(i.toString) }.mkString(",") )
        .append(",")
        .append(gammaTitles)
        .append(",")
        .append(lambdaTitles)
        .append("\n")

      breeze.linalg.csvwrite(outputFile, resMat, separator = ',')
      pw.close()
    }
  }
}
