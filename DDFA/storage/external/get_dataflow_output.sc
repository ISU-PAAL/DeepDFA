import io.joern.dataflowengineoss.passes.reachingdef.ReachingDefPass
import io.joern.dataflowengineoss.passes.reachingdef.ReachingDefProblem
import io.joern.dataflowengineoss.passes.reachingdef.ReachingDefTransferFunction
import io.joern.dataflowengineoss.passes.reachingdef.ReachingDefFlowGraph
import scala.collection.immutable.HashMap
import better.files.File

// https://stackoverflow.com/a/55967256/8999671
def toJson(query: Any): String = query match {
   case m: Map[String, Any] => s"{${m.map(toJson(_)).mkString(",")}}"
   case t: (String, Any) => s""""${t._1}":${toJson(t._2)}"""
   case ss: Seq[Any] => s"""[${ss.map(toJson(_)).mkString(",")}]"""
   case s: String => s""""$s""""
   case null => "null"
   case _ => query.toString
}

/**
  * Instantiate reaching definition problem and print the solution
  *
  * Run with:
  * joern --script storage/external/get_dataflow_output.scala --params filename=x42/c/X42.c
  */
@main def exec(filename: String, cache: Boolean = true) = {
    val summaryFile = File(s"${filename}.dataflow.summary.json")
    if (!summaryFile.exists || !cache) {
        try {
            val cpgFile = File(filename + ".cpg.bin")
            if (cpgFile.exists) {
                println(s"Loading CPG from $cpgFile")
                importCpg(cpgFile.toString)
            }
            else {
                println(s"Exporting CPG to $cpgFile")
                importCode.c(filename)
            }

            val method2df = cpg.method.filter(m => m.filename != "<empty>" && m.name != "<global>").map(m => {
                val problem = ReachingDefProblem.create(m);
                val solution = new DataFlowSolver().calculateMopSolutionForwards(problem);
                val transferFunction = problem.transferFunction.asInstanceOf[ReachingDefTransferFunction];
                val numberToNode = problem.flowGraph.asInstanceOf[ReachingDefFlowGraph].numberToNode;
                val df = HashMap(
                    "problem.gen" -> transferFunction.gen.map(kv => (kv._1.id.toString, kv._2.toList.sorted.map(numberToNode).map(_.id))).toSeq.sortBy(_._1).toMap,
                    "problem.kill" -> transferFunction.kill.map(kv => (kv._1.id.toString, kv._2.toList.sorted.map(numberToNode).map(_.id))).toSeq.sortBy(_._1).toMap,
                    "solution.in" -> solution.in.map(kv => (kv._1.id.toString, kv._2.toList.sorted.map(numberToNode).map(_.id))).toSeq.sortBy(_._1).toMap,
                    "solution.out" -> solution.in.map(kv => (kv._1.id.toString, kv._2.toList.sorted.map(numberToNode).map(_.id))).toSeq.sortBy(_._1).toMap,
                );
                (m.name, df)
            }).toMap
            toJson(method2df) |> s"${filename}.dataflow.json"
            println("Done exporting dataflow")
        }
        catch {
            case e : Exception => {
                println("Error during execution:")
                e.printStackTrace()
            }
        }
        finally {
            try {
                delete
            }
            catch {
                case e : RuntimeException => println(s"Error deleting project: ${e.getMessage}")
            }
        }
    }
    else {
        println(f"result is cached $filename")
    }
}
