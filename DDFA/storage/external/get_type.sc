import scala.collection.mutable.HashSet

// Get declaration of typename, resolving aliases
def trueTypeDecl(tn: String): Traversal[TypeDecl] = {
    val aliasIt = cpg.typeDecl.name(tn).aliasTypeFullName.dedup
    if (aliasIt.hasNext) {
        val aliasedType = aliasIt.head
        if (aliasedType.startsWith("anonymous_type_")) {
            val anonNum = aliasedType.substring("anonymous_type_".length).toInt
            cpg.typeDecl.name("").filename(cpg.typeDecl.name(tn).filename.head).sortBy(_.order).drop(anonNum).take(1)
        }
        else {
            cpg.typeDecl.name(aliasedType)
        }
        
    }
    else {
        cpg.typeDecl.name(tn)
    }
}

// map TypeDecl to its most grandchild leaf types.
// Leaf types are external types or internal types without members.
def mapToMemberTypes(allT: List[TypeDecl], seen: HashSet[String] = HashSet()): List[String] = {
    seen ++= allT.name
    var toReturn = allT.external.name.l
    // DEBUG: do not filter out external fields
    val t = allT
    //val t = allT.filter(!_.isExternal).l
    if (!t.member.hasNext) {
        toReturn = toReturn ::: t.name.l
    }
    else {
        val memberTypeNames = t.member.typeFullName.filterNot(seen).l
        seen ++= memberTypeNames
        toReturn = toReturn ::: memberTypeNames
            .map(m => trueTypeDecl(m).l)
            .flatMap(mapToMemberTypes(_: List[TypeDecl], seen)).dedup.l
    }
    toReturn
}

@main def exec(cpgName: String = null, rootType: String, outFile: String = null) = {
    val trueType = trueTypeDecl(rootType).dedup.l
    val memberTypes = mapToMemberTypes(trueType).sorted.dedup.l
    println(s"Exporting ${memberTypes.length} types")
    val outFileName = outFile match {
        case null => s"memberTypes_$rootType.txt"
        case s => s
    }
    memberTypes.mkString("\n") |> outFileName
}
