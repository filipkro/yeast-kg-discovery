from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, XSD
from pathlib import Path


RIMBO = Namespace("http://rimbo.project-genesis.io#")
KG = Namespace("http://example.org/yeast/kg/")
CHEBI = Namespace("http://purl.obolibrary.org/obo/")
DECREASES_CHEMICAL_STRESS_RESISTANCE = (
    KG.hasChemStressResistance_Decreased
)

# =============================================================================
# KG loading / saving
# =============================================================================

def load_kg(path: str | Path) -> Graph:
    """
    Load an RDF knowledge graph from a Turtle file.
    """
    graph = Graph()
    graph.parse(str(path), format="turtle")
    return graph


def save_kg(
    graph: Graph,
    path: str | Path,
) -> None:
    """
    Save an RDF knowledge graph as Turtle.
    """
    graph.serialize(destination=str(path), format="turtle")


# =============================================================================
# KG URI utilities
# =============================================================================

def yeast_gene(orf_name: str) -> URIRef:
    """
    Return the KG URI for a yeast gene.

    Example:
        yeast_gene("YBR123C")
        -> kg:YBR123C
    """
    return KG[orf_name]


def chemical_uri(chemical_id: str) -> URIRef:
    """
    Convert a chemical identifier to an RDF URI.

    Examples:
        CHEBI:15377
        CHEBI_15377

    are converted to:
        chebi:CHEBI_15377
    """

    if chemical_id.startswith("CHEBI:"):
        local_id = chemical_id.split(":", 1)[1]
        return CHEBI[f"CHEBI_{local_id}"]

    if chemical_id.startswith("CHEBI_"):
        return CHEBI[chemical_id]

    # If the chemical is already represented as a URI-like value,
    # this can be changed according to the KG's identifier scheme.
    return KG[chemical_id]


# =============================================================================
# Hypothesis / triple construction
# =============================================================================

def make_hypothesis_triple(
    orf_name: str,
    chemical_id: str,
    predicate: URIRef = DECREASES_CHEMICAL_STRESS_RESISTANCE,
) -> tuple[URIRef, URIRef, URIRef]:
    """
    Construct the RDF triple corresponding to a biological hypothesis.

    Example:

        kg:YBR123C
            kg:decreasesChemicalStressResistance
            chebi:CHEBI_15377
    """

    subject = yeast_gene(orf_name)
    obj = chemical_uri(chemical_id)

    return subject, URIRef(predicate), obj


# =============================================================================
# RIMBO KnowledgeGraph representation
# =============================================================================

def create_knowledge_graph(
    graph: Graph,
    kg_id: str,
    file_path: str | Path,
) -> tuple[URIRef, URIRef]:
    """
    Create a RIMBO KnowledgeGraph and its Turtle file representation.

    The resulting structure is:

        kg:yeast_kg_v0
            a rimbo:KnowledgeGraph ;
            rimbo:isImplementedAs kg:yeast_kg_v0_file .

        kg:yeast_kg_v0_file
            a rimbo:Turtle ;
            rimbo:filePath "/path/to/yeast_kg.ttl" .

    Returns:
        (knowledge_graph_uri, file_representation_uri)
    """

    kg = KG[kg_id]
    representation = KG[f"{kg_id}_file"]

    # KnowledgeGraph
    graph.add((kg, RDF.type, RIMBO.KnowledgeGraph))

    # File representation
    graph.add((representation, RDF.type, RIMBO.Turtle))

    graph.add((representation, RIMBO.filePath, Literal(str(file_path))))

    # KnowledgeGraph -> file representation
    graph.add((kg, RIMBO.isImplementedAs, representation))

    return kg, representation


# =============================================================================
# RIMBO revision creation
# =============================================================================

def create_revision(
    graph: Graph,
    revision_id: str,
    original_kg: URIRef,
    original_kg_representation: URIRef,
    orf_name: str,
    chemical_id: str,
    p_value: float,
    fdr: float,
    supports_hypothesis: bool,
    predicate: URIRef = DECREASES_CHEMICAL_STRESS_RESISTANCE,
) -> URIRef:
    """
    Create a RIMBO revision describing an experimentally tested hypothesis.

    Supported hypothesis:

        Revision
        ├── hasChange -> TripleInsertion
        ├── hasReason -> ExperimentalEvidence
        └── hasPatch  -> KGTriplePatch
                             └── patchTo -> original KG file

    Unsupported hypothesis:

        UnsupportedRevision
        ├── hasChange -> TripleInsertion
        └── hasReason -> ExperimentalEvidence

    No patch is created for an unsupported hypothesis.

    Returns:
        URI of the created revision.
    """

    revision = KG[revision_id]

    # -------------------------------------------------------------------------
    # Construct the hypothesis triple
    # -------------------------------------------------------------------------

    subject, predicate, obj = make_hypothesis_triple(
        orf_name=orf_name,
        chemical_id=chemical_id,
        predicate=predicate,
    )

    # -------------------------------------------------------------------------
    # Revision type
    # -------------------------------------------------------------------------

    if supports_hypothesis:
        revision_type = RIMBO.Revision
    else:
        revision_type = RIMBO.UnsupportedRevision

    graph.add((revision, RDF.type, revision_type))

    # Revision -> original KG
    graph.add((revision, RIMBO.revisionTo, original_kg))

    # -------------------------------------------------------------------------
    # Change
    # -------------------------------------------------------------------------

    change = KG[f"{revision_id}_change"]

    graph.add((change, RDF.type, RIMBO.TripleInsertion))

    graph.add((revision, RIMBO.hasChange, change))

    # The actual RDF triple being proposed/validated
    graph.add((change, RIMBO.hasSubject, subject))

    graph.add((change, RIMBO.hasPredicate, predicate))

    graph.add((change, RIMBO.hasObject, obj))

    # -------------------------------------------------------------------------
    # Experimental evidence
    # -------------------------------------------------------------------------

    evidence = KG[f"{revision_id}_evidence"]

    graph.add((evidence, RDF.type, RIMBO.ExperimentalEvidence))

    graph.add((revision, RIMBO.hasReason, evidence))

    graph.add((evidence, RIMBO.pValue, Literal(p_value, datatype=XSD.double)))

    graph.add((evidence, RIMBO.fdr, Literal(fdr, datatype=XSD.double)))

    graph.add((evidence, RIMBO.supportsHypothesis,
               Literal(supports_hypothesis, datatype=XSD.boolean)))

    # -------------------------------------------------------------------------
    # Patch
    # -------------------------------------------------------------------------

    # Only experimentally supported hypotheses become KG patches.
    if supports_hypothesis:

        patch = KG[f"{revision_id}_patch"]

        graph.add((patch, RDF.type, RIMBO.KGTriplePatch))

        graph.add((revision, RIMBO.hasPatch, patch))

        # Construct a one-triple Turtle patch.
        turtle_string = (f"<{subject}> <{predicate}> <{obj}> .")

        graph.add((patch, RIMBO.stringRepresentation, Literal(turtle_string)))

        # IMPORTANT:
        # patchTo points to the actual file representation, not the
        # Turtle class.
        graph.add((patch, RIMBO.patchTo, original_kg_representation))

    return revision


# =============================================================================
# Patch application
# =============================================================================

def apply_patch(
    graph: Graph,
    patch: URIRef,
) -> None:
    """
    Apply a KGTriplePatch to an RDF graph.

    The patch is expected to contain exactly one stringRepresentation
    containing valid Turtle.
    """

    patch_strings = list(graph.objects(patch, RIMBO.stringRepresentation))

    if len(patch_strings) != 1:
        raise ValueError(
            f"Patch {patch} must have exactly one "
            f"rimbo:stringRepresentation; "
            f"found {len(patch_strings)}."
        )

    turtle_string = str(patch_strings[0])

    patch_graph = Graph()

    patch_graph.parse(data=turtle_string, format="turtle")

    for triple in patch_graph:
        graph.add(triple)


# =============================================================================
# Finding revisions
# =============================================================================

def get_supported_revisions(
    metadata_graph: Graph,
    original_kg: URIRef,
) -> list[URIRef]:
    """
    Get supported revisions that are revisions of original_kg.

    UnsupportedRevision instances are excluded.
    """

    revisions = []

    for revision in metadata_graph.subjects(RIMBO.revisionTo, original_kg):

        # Unsupported revisions are deliberately excluded.
        if (revision, RDF.type, RIMBO.UnsupportedRevision) in metadata_graph:
            continue

        revisions.append(revision)

    return revisions


# =============================================================================
# Finding patches
# =============================================================================

def get_patches_for_revision(
    metadata_graph: Graph,
    revision: URIRef,
) -> list[URIRef]:
    """
    Return all patches associated with a revision.
    """

    return list(metadata_graph.objects(revision, RIMBO.hasPatch))


# =============================================================================
# Reconstructing the KG
# =============================================================================

def reconstruct_kg(
    original_kg_path: str | Path,
    metadata_graph: Graph,
    original_kg: URIRef,
    original_kg_representation: URIRef,
) -> Graph:
    """
    Reconstruct an updated KG by starting from the original KG and
    applying patches from supported revisions.

    Unsupported hypotheses are retained in the metadata graph but
    contribute no patch and therefore do not modify the reconstructed KG.
    """

    # Start from the original KG.
    kg = load_kg(original_kg_path)

    # Find revisions associated with this KG.
    revisions = get_supported_revisions(
        metadata_graph=metadata_graph,
        original_kg=original_kg,
    )

    for revision in revisions:

        patches = get_patches_for_revision(metadata_graph=metadata_graph,
                                           revision=revision)

        for patch in patches:

            # Make sure this patch actually applies to the representation
            # we are reconstructing.
            patch_targets = list(metadata_graph.objects(patch, RIMBO.patchTo))

            if original_kg_representation not in patch_targets:
                continue

            apply_patch(graph=kg, patch=patch)

    return kg


# =============================================================================
# Optional: inspect what was added by a revision
# =============================================================================

def get_change_triple(
    metadata_graph: Graph,
    revision: URIRef,
) -> tuple[URIRef, URIRef, URIRef]:
    """
    Extract the RDF triple represented by a revision's change.
    """

    changes = list(metadata_graph.objects(revision, RIMBO.hasChange))

    if len(changes) != 1:
        raise ValueError(
            f"Expected exactly one change for {revision}, "
            f"found {len(changes)}."
        )

    change = changes[0]

    subjects = list(metadata_graph.objects(change, RIMBO.hasSubject))

    predicates = list(metadata_graph.objects(change, RIMBO.hasPredicate))

    objects = list(metadata_graph.objects(change, RIMBO.hasObject))

    if len(subjects) != 1:
        raise ValueError(f"Expected exactly one subject for {change}.")

    if len(predicates) != 1:
        raise ValueError(f"Expected exactly one predicate for {change}.")

    if len(objects) != 1:
        raise ValueError(f"Expected exactly one object for {change}.")

    return (subjects[0], predicates[0], objects[0])


def print_revision(
    graph: Graph,
    revision: URIRef | str,
) -> None:
    """
    Print a human-readable overview of a RIMBO revision.

    Useful both for inspecting the provenance graph and for sanity
    checking that a revision has been constructed correctly.
    """
    # -------------------------------------------------------------------------
    # Resolve revision URI
    # -------------------------------------------------------------------------

    if isinstance(revision, str):
        revision = URIRef(revision)

    print()
    print("=" * 80)
    print("RIMBO REVISION")
    print("=" * 80)

    print(f"Revision: {revision}")

    # -------------------------------------------------------------------------
    # Revision type
    # -------------------------------------------------------------------------

    types = list(graph.objects(revision, RDF.type))

    print()
    print("Type")
    print("-" * 80)

    if types:
        for revision_type in types:
            print(f"  {revision_type}")
    else:
        print("  ⚠ No rdf:type found")

    is_unsupported = (revision, RDF.type, RIMBO.UnsupportedRevision) in graph

    is_supported = (revision, RDF.type, RIMBO.Revision) in graph

    if is_unsupported:
        print("  Status: UNSUPPORTED HYPOTHESIS")
    elif is_supported:
        print("  Status: SUPPORTED HYPOTHESIS")
    else:
        print("  ⚠ Status could not be determined")

    # -------------------------------------------------------------------------
    # KG being revised
    # -------------------------------------------------------------------------

    print()
    print("Revises")
    print("-" * 80)

    original_kgs = list(graph.objects(revision, RIMBO.revisionTo))

    if not original_kgs:
        print("  ⚠ No revisionTo relation found")

    for kg in original_kgs:

        print(f"  KnowledgeGraph: {kg}")

        kg_types = list(graph.objects(kg, RDF.type))

        for kg_type in kg_types:
            print(f"    Type: {kg_type}")

        # -------------------------------------------------------------
        # KG file representation
        # -------------------------------------------------------------

        representations = list(graph.objects(kg, RIMBO.isImplementedAs))

        if not representations:
            print("    ⚠ No file representation found")

        for representation in representations:

            print(f"    Representation: {representation}")

            representation_types = list(graph.objects(representation, RDF.type))

            for representation_type in representation_types:
                print(f"      Type: {representation_type}")

            paths = list(graph.objects(representation, RIMBO.filePath))

            if paths:
                for path in paths:
                    print(f"      File path: {path}")
            else:
                print("      ⚠ No filePath found")

    # -------------------------------------------------------------------------
    # Changes
    # -------------------------------------------------------------------------

    print()
    print("Changes")
    print("-" * 80)

    changes = list(graph.objects(revision, RIMBO.hasChange))

    if not changes:
        print("  ⚠ No changes found")

    for i, change in enumerate(changes, start=1):

        print(f"  Change {i}: {change}")

        change_types = list(graph.objects(change, RDF.type))

        for change_type in change_types:
            print(f"    Type: {change_type}")

        subjects = list(graph.objects(change, RIMBO.hasSubject))

        predicates = list(graph.objects(change, RIMBO.hasPredicate))

        objects = list(graph.objects(change, RIMBO.hasObject))

        print("    Triple:")

        print(
            f"      Subject:   {subjects[0]}"
            if subjects
            else "      Subject:   ⚠ MISSING"
        )

        print(
            f"      Predicate: {predicates[0]}"
            if predicates
            else "      Predicate: ⚠ MISSING"
        )

        print(
            f"      Object:    {objects[0]}"
            if objects
            else "      Object:    ⚠ MISSING"
        )

    # -------------------------------------------------------------------------
    # Experimental evidence
    # -------------------------------------------------------------------------

    print()
    print("Experimental Evidence")
    print("-" * 80)

    reasons = list(graph.objects(revision, RIMBO.hasReason))

    if not reasons:
        print("  ⚠ No reason/evidence found")

    for i, evidence in enumerate(reasons, start=1):

        print(f"  Evidence {i}: {evidence}")

        evidence_types = list(graph.objects(evidence, RDF.type))

        for evidence_type in evidence_types:
            print(f"    Type: {evidence_type}")

        p_values = list(graph.objects(evidence, RIMBO.pValue))

        fdrs = list(graph.objects(evidence, RIMBO.fdr))

        support = list(graph.objects(evidence, RIMBO.supportsHypothesis))

        print(
            f"    p-value: "
            f"{p_values[0]}"
            if p_values
            else "    p-value: ⚠ MISSING"
        )

        print(
            f"    FDR: "
            f"{fdrs[0]}"
            if fdrs
            else "    FDR: ⚠ MISSING"
        )

        print(
            f"    Supports hypothesis: "
            f"{support[0]}"
            if support
            else "    Supports hypothesis: ⚠ MISSING"
        )

    # -------------------------------------------------------------------------
    # Patch
    # -------------------------------------------------------------------------

    print()
    print("Patch")
    print("-" * 80)

    patches = list(graph.objects(revision, RIMBO.hasPatch))

    if not patches:
        if is_unsupported:
            print("  No patch (expected for unsupported hypothesis)")
        else:
            print("  ⚠ No patch found")

    for i, patch in enumerate(patches, start=1):

        print(f"  Patch {i}: {patch}")

        patch_types = list(graph.objects(patch, RDF.type))

        for patch_type in patch_types:
            print(f"    Type: {patch_type}")

        patch_targets = list(graph.objects(patch, RIMBO.patchTo))

        if patch_targets:
            for target in patch_targets:
                print(f"    Patch applies to: {target}")
        else:
            print("    ⚠ No patchTo relation found")

        representations = list(graph.objects(patch, RIMBO.stringRepresentation))

        if representations:
            print("    Turtle patch:")
            for representation in representations:
                print(f"      {representation}")
        else:
            print("    ⚠ No stringRepresentation found")

    # -------------------------------------------------------------------------
    # Basic consistency checks
    # -------------------------------------------------------------------------

    print()
    print("Sanity checks")
    print("-" * 80)

    problems = []

    if not original_kgs:
        problems.append("Revision has no revisionTo relation.")

    if not changes:
        problems.append("Revision has no change.")

    if not reasons:
        problems.append("Revision has no experimental evidence.")

    if is_unsupported and patches:
        problems.append("UnsupportedRevision has a patch.")

    if is_supported and not patches:
        problems.append("Supported Revision has no patch.")

    for patch in patches:

        targets = list(graph.objects(patch, RIMBO.patchTo))

        if not targets:
            problems.append(f"Patch {patch} has no patchTo target.")

    if problems:
        print("  ⚠ Problems found:")
        for problem in problems:
            print(f"    - {problem}")
    else:
        print("  ✓ No obvious structural problems found.")

    print()
    print("=" * 80)
    print()