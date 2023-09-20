import better.files.File
import io.joern.dataflowengineoss.passes.reachingdef.ReachingDefPass
import io.joern.dataflowengineoss.passes.reachingdef.ReachingDefProblem
import io.joern.dataflowengineoss.passes.reachingdef.ReachingDefTransferFunction
import io.joern.dataflowengineoss.passes.reachingdef.ReachingDefFlowGraph
import io.joern.dataflowengineoss.passes.reachingdef.DataFlowSolver
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
@main def exec(filename: String, runOssDataflow: Boolean = true, exportJson: Boolean = true, exportCpg: Boolean = true, exportDataflow: Boolean = true, deleteAfter: Boolean = true, overwrite: Boolean = false) = {
   val cpgFile = File(filename + ".cpg.bin")
   if (cpgFile.exists) {
      println(s"Loading CPG from $cpgFile")
      importCpg(cpgFile.toString)
   }
   else {
      println(s"Exporting CPG to $cpgFile")
      importCode(filename)
      if (runOssDataflow) {
         run.ossdataflow
      }
   }
   if (exportCpg) {
      save
      val outputFilename = filename + ".cpg.bin"
      val outputFile = File(outputFilename)
      if (!outputFile.exists) {
         println(s"Exporting CPG to $outputFilename")
         File(project.path + "/cpg.bin").copyTo(outputFile, overwrite=true)
      }
   }
   if (exportJson) {
      val nodeOutputFilename = filename + ".nodes.json"
      val edgeOutputFilename = filename + ".edges.json"
      if (!File(nodeOutputFilename).exists || !File(edgeOutputFilename).exists) {
         println(s"Exporting JSON to $nodeOutputFilename $edgeOutputFilename")
         cpg.graph.E.map(node=>List(node.inNode.id, node.outNode.id, node.label, node.propertiesMap.get("VARIABLE"))).toJson |> edgeOutputFilename
         cpg.graph.V.map(node=>node).toJson |> nodeOutputFilename
      }
   }
   if (exportDataflow) {
      val dataflowFilename = filename + ".dataflow.json"
      if (!File(dataflowFilename).exists) {
         println(s"Exporting dataflow solution to $dataflowFilename")
         val method2df = cpg.method.filter(m => m.filename != "<empty>" && m.name != "<global>").map(m => {
            val problem = ReachingDefProblem.create(m);
            val solution = new DataFlowSolver().calculateMopSolutionForwards(problem);
            val transferFunction = problem.transferFunction.asInstanceOf[ReachingDefTransferFunction];
            val numberToNode = problem.flowGraph.asInstanceOf[ReachingDefFlowGraph].numberToNode;
            val df = HashMap(
               "problem.gen" -> transferFunction.gen.map(kv => (kv._1.id.toString, kv._2.toList.sorted.map(numberToNode).map(_.id))).toSeq.sortBy(_._1).toMap,
               "problem.kill" -> transferFunction.kill.map(kv => (kv._1.id.toString, kv._2.toList.sorted.map(numberToNode).map(_.id))).toSeq.sortBy(_._1).toMap,
               "solution.in" -> solution.in.map(kv => (kv._1.id.toString, kv._2.toList.sorted.map(numberToNode).map(_.id))).toSeq.sortBy(_._1).toMap,
               "solution.out" -> solution.out.map(kv => (kv._1.id.toString, kv._2.toList.sorted.map(numberToNode).map(_.id))).toSeq.sortBy(_._1).toMap,
            );
            (m.name, df)
         }).toMap
         toJson(method2df) |> dataflowFilename
      }
   }
   if (deleteAfter) {
      delete
   }
}

