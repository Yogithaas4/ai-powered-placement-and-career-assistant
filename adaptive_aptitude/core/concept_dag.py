"""
concept_dag.py
--------------
Builds and manages a Directed Acyclic Graph of concepts/topics.
Nodes = concepts (topic + subtopic)
Edges = prerequisite relationships (A → B means "learn A before B")

Used by the question selector to:
  - Avoid testing advanced concepts before prerequisites are mastered
  - Surface foundational gaps when a student struggles
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import json


@dataclass
class ConceptNode:
    concept_id: str          # e.g. "CN::OSI_Model"
    subject: str             # e.g. "Computer Networks" (== canonical_subject)
    topic: str               # e.g. "OSI Model"
    subtopic: str            # e.g. "Layer Functions"
    prerequisites: List[str] = field(default_factory=list)   # concept_ids that must come first
    dependents: List[str]   = field(default_factory=list)    # concept_ids that build on this
    # practice_category is the broad, student-facing bucket (e.g. "Core CS
    # (Systems & Theory)") that `subject` rolls up into -- optional so old
    # callers that never set it (build_default_dag) keep working unchanged.
    practice_category: Optional[str] = None


class ConceptDAG:
    """
    Directed Acyclic Graph of concepts.
    Edges point FROM prerequisite TO dependent.
    """

    def __init__(self):
        self.nodes: Dict[str, ConceptNode] = {}

    # ── Building ───────────────────────────────────────────────────────────

    def add_concept(self, concept_id: str, subject: str, topic: str, subtopic: str,
                     practice_category: Optional[str] = None):
        if concept_id not in self.nodes:
            self.nodes[concept_id] = ConceptNode(
                concept_id=concept_id,
                subject=subject,
                topic=topic,
                subtopic=subtopic,
                practice_category=practice_category
            )

    def add_prerequisite(self, prereq_id: str, dependent_id: str):
        """prereq_id must be learned before dependent_id."""
        if prereq_id in self.nodes and dependent_id in self.nodes:
            if dependent_id not in self.nodes[prereq_id].dependents:
                self.nodes[prereq_id].dependents.append(dependent_id)
            if prereq_id not in self.nodes[dependent_id].prerequisites:
                self.nodes[dependent_id].prerequisites.append(prereq_id)

    # ── Querying ───────────────────────────────────────────────────────────

    def get_prerequisites(self, concept_id: str, deep: bool = False) -> List[str]:
        """Return direct (or all transitive) prerequisites of a concept."""
        if concept_id not in self.nodes:
            return []
        if not deep:
            return self.nodes[concept_id].prerequisites
        # BFS for all ancestors
        visited, queue = set(), list(self.nodes[concept_id].prerequisites)
        while queue:
            curr = queue.pop(0)
            if curr not in visited:
                visited.add(curr)
                queue.extend(self.nodes[curr].prerequisites)
        return list(visited)

    def get_concepts_by_subject(self, subject: str) -> List[str]:
        return [nid for nid, n in self.nodes.items() if n.subject == subject]

    def get_concepts_by_practice_category(self, practice_category: str) -> List[str]:
        """Real sessions are scoped by this (5 broad buckets), not by
        canonical_subject (22) -- a "Core CS" session must be able to pick
        questions from Computer Networks, OS, Databases, Digital Logic,
        etc. all in one pool. canonical_subject is still tracked per
        concept for mastery rollups/dashboard reporting, it's just not the
        session-scoping key."""
        return [nid for nid, n in self.nodes.items() if n.practice_category == practice_category]

    def get_unlocked_concepts(self, subject: str, mastery: Dict[str, float],
                               mastery_threshold: float = 0.6) -> List[str]:
        """Subject-scoped variant, kept for existing callers (build_default_dag
        callers, e.g. the live CN-only prototype)."""
        return self._filter_unlocked(self.get_concepts_by_subject(subject), mastery, mastery_threshold)

    def get_unlocked_concepts_in(self, concept_ids: List[str], mastery: Dict[str, float],
                                  mastery_threshold: float = 0.6) -> List[str]:
        """Scope-agnostic variant: pass any candidate set (e.g. the output
        of get_concepts_by_practice_category) and get back the subset whose
        prerequisites are sufficiently mastered."""
        return self._filter_unlocked(concept_ids, mastery, mastery_threshold)

    def _filter_unlocked(self, concept_ids: List[str], mastery: Dict[str, float],
                          mastery_threshold: float) -> List[str]:
        unlocked = []
        for cid in concept_ids:
            prereqs = self.get_prerequisites(cid, deep=False)
            if all(mastery.get(p, 0.0) >= mastery_threshold for p in prereqs):
                unlocked.append(cid)
        return unlocked

    def topological_order(self, subject: Optional[str] = None) -> List[str]:
        """Return concepts in learning order (prerequisites first)."""
        nodes = self.get_concepts_by_subject(subject) if subject else list(self.nodes.keys())
        in_degree = {n: 0 for n in nodes}
        for nid in nodes:
            for dep in self.nodes[nid].dependents:
                if dep in in_degree:
                    in_degree[dep] += 1

        queue = [n for n in nodes if in_degree[n] == 0]
        order = []
        while queue:
            curr = queue.pop(0)
            order.append(curr)
            for dep in self.nodes[curr].dependents:
                if dep in in_degree:
                    in_degree[dep] -= 1
                    if in_degree[dep] == 0:
                        queue.append(dep)
        return order

    # ── Serialization ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {nid: {
            "subject": n.subject, "topic": n.topic, "subtopic": n.subtopic,
            "prerequisites": n.prerequisites, "dependents": n.dependents,
            "practice_category": n.practice_category
        } for nid, n in self.nodes.items()}

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ConceptDAG":
        with open(path) as f:
            data = json.load(f)
        dag = cls()
        for cid, info in data.items():
            dag.nodes[cid] = ConceptNode(
                concept_id=cid,
                subject=info["subject"], topic=info["topic"], subtopic=info["subtopic"],
                prerequisites=info["prerequisites"], dependents=info["dependents"],
                practice_category=info.get("practice_category")
            )
        return dag


# ── Default DAG for CS subjects ────────────────────────────────────────────

def build_default_dag() -> ConceptDAG:
    """
    Build a hand-crafted prerequisite graph for the 8 CS subjects.
    This encodes known learning dependencies.
    """
    dag = ConceptDAG()

    # ── Computer Networks ──────────────────────────────────────────────────
    cn_concepts = [
        ("CN::Physical_Encoding",       "Computer Networks", "Physical Layer",    "Encoding"),
        ("CN::Physical_Media",          "Computer Networks", "Physical Layer",    "Transmission Media"),
        ("CN::Physical_Bandwidth",      "Computer Networks", "Physical Layer",    "Bandwidth & Data Rate"),
        ("CN::DLL_Framing",             "Computer Networks", "Data Link Layer",   "Framing & Protocols"),
        ("CN::DLL_ErrorDetection",      "Computer Networks", "Data Link Layer",   "Error Detection"),
        ("CN::DLL_FlowControl",         "Computer Networks", "Data Link Layer",   "Flow Control"),
        ("CN::DLL_MAC",                 "Computer Networks", "Data Link Layer",   "MAC Protocols"),
        ("CN::OSI_Model",               "Computer Networks", "OSI Model",         "Layer Functions"),
        ("CN::Network_IP",              "Computer Networks", "Network Layer",     "IP Protocol"),
        ("CN::Network_Addressing",      "Computer Networks", "Network Layer",     "IP Addressing"),
        ("CN::Network_Routing",         "Computer Networks", "Network Layer",     "Routing"),
        ("CN::Transport_TCP",           "Computer Networks", "Transport Layer",   "TCP"),
        ("CN::Transport_UDP",           "Computer Networks", "Transport Layer",   "UDP"),
        ("CN::Application_DNS",         "Computer Networks", "Application Layer", "DNS & HTTP"),
        ("CN::Security",                "Computer Networks", "Network Security",  "Encryption & Sniffing"),
        ("CN::Topology",                "Computer Networks", "Network Topology",  "Topology Types"),
        ("CN::Devices",                 "Computer Networks", "Network Devices",   "Switches & Hubs"),
    ]
    for args in cn_concepts:
        dag.add_concept(*args)

    cn_prereqs = [
        ("CN::Physical_Encoding",   "CN::DLL_Framing"),
        ("CN::Physical_Encoding",   "CN::DLL_ErrorDetection"),
        ("CN::Physical_Bandwidth",  "CN::DLL_FlowControl"),
        ("CN::DLL_Framing",         "CN::DLL_FlowControl"),
        ("CN::DLL_ErrorDetection",  "CN::DLL_FlowControl"),
        ("CN::DLL_FlowControl",     "CN::DLL_MAC"),
        ("CN::OSI_Model",           "CN::Network_IP"),
        ("CN::DLL_MAC",             "CN::Network_IP"),
        ("CN::Network_IP",          "CN::Network_Addressing"),
        ("CN::Network_Addressing",  "CN::Network_Routing"),
        ("CN::Network_Routing",     "CN::Transport_TCP"),
        ("CN::Network_Routing",     "CN::Transport_UDP"),
        ("CN::Transport_TCP",       "CN::Application_DNS"),
        ("CN::Transport_TCP",       "CN::Security"),
    ]
    for p, d in cn_prereqs:
        dag.add_prerequisite(p, d)

    # ── Operating System ───────────────────────────────────────────────────
    os_concepts = [
        ("OS::Process_States",      "Operating System", "Process Management", "Process States"),
        ("OS::Context_Switch",      "Operating System", "Process Management", "Context Switching"),
        ("OS::Process_Creation",    "Operating System", "Process Management", "Process Creation"),
        ("OS::Scheduling_Algo",     "Operating System", "CPU Scheduling",     "Scheduling Algorithms"),
        ("OS::Scheduling_Metrics",  "Operating System", "CPU Scheduling",     "Scheduling Metrics"),
        ("OS::Memory_Paging",       "Operating System", "Memory Management",  "Paging"),
        ("OS::Memory_Virtual",      "Operating System", "Memory Management",  "Virtual Memory"),
        ("OS::Memory_Alloc",        "Operating System", "Memory Management",  "Memory Allocation"),
        ("OS::Deadlock_Detection",  "Operating System", "Deadlocks",          "Deadlock Detection"),
        ("OS::Deadlock_Prevention", "Operating System", "Deadlocks",          "Deadlock Prevention"),
        ("OS::Sync_Semaphore",      "Operating System", "Synchronization",    "Semaphores"),
        ("OS::Sync_Problems",       "Operating System", "Synchronization",    "Classic Problems"),
        ("OS::Sync_Race",           "Operating System", "Synchronization",    "Race Conditions"),
        ("OS::FileSystem",          "Operating System", "File Systems",       "File Allocation"),
        ("OS::Disk_Scheduling",     "Operating System", "I/O Management",     "Disk Scheduling"),
        ("OS::Protection",          "Operating System", "Protection & Security", "Protection"),
    ]
    for args in os_concepts:
        dag.add_concept(*args)

    os_prereqs = [
        ("OS::Process_States",     "OS::Context_Switch"),
        ("OS::Process_States",     "OS::Process_Creation"),
        ("OS::Process_States",     "OS::Scheduling_Algo"),
        ("OS::Scheduling_Algo",    "OS::Scheduling_Metrics"),
        ("OS::Memory_Alloc",       "OS::Memory_Paging"),
        ("OS::Memory_Paging",      "OS::Memory_Virtual"),
        ("OS::Process_States",     "OS::Deadlock_Detection"),
        ("OS::Deadlock_Detection", "OS::Deadlock_Prevention"),
        ("OS::Sync_Race",          "OS::Sync_Semaphore"),
        ("OS::Sync_Semaphore",     "OS::Sync_Problems"),
        ("OS::Memory_Paging",      "OS::FileSystem"),
        ("OS::Process_States",     "OS::Protection"),
    ]
    for p, d in os_prereqs:
        dag.add_prerequisite(p, d)

    # ── Digital Logic ──────────────────────────────────────────────────────
    dl_concepts = [
        ("DL::Number_Base",         "Digital Logic", "Number Systems",        "Base Conversion"),
        ("DL::Number_Signed",       "Digital Logic", "Number Systems",        "Signed Numbers"),
        ("DL::Bool_Laws",           "Digital Logic", "Boolean Algebra",       "Boolean Laws"),
        ("DL::Bool_Simplify",       "Digital Logic", "Boolean Algebra",       "Simplification"),
        ("DL::Comb_Gates",          "Digital Logic", "Combinational Circuits","Logic Gates"),
        ("DL::Comb_MUX",            "Digital Logic", "Combinational Circuits","Multiplexers"),
        ("DL::Comb_Adders",         "Digital Logic", "Combinational Circuits","Adders & Subtractors"),
        ("DL::Seq_FlipFlop",        "Digital Logic", "Sequential Circuits",   "Flip-Flops"),
        ("DL::Seq_Counter",         "Digital Logic", "Sequential Circuits",   "Counters"),
        ("DL::Seq_FSM",             "Digital Logic", "Sequential Circuits",   "State Machines"),
        ("DL::IEEE754",             "Digital Logic", "IEEE 754",              "Floating Point Representation"),
    ]
    for args in dl_concepts:
        dag.add_concept(*args)

    dl_prereqs = [
        ("DL::Number_Base",   "DL::Number_Signed"),
        ("DL::Number_Base",   "DL::IEEE754"),
        ("DL::Bool_Laws",     "DL::Bool_Simplify"),
        ("DL::Bool_Simplify", "DL::Comb_Gates"),
        ("DL::Comb_Gates",    "DL::Comb_MUX"),
        ("DL::Comb_Gates",    "DL::Comb_Adders"),
        ("DL::Comb_Gates",    "DL::Seq_FlipFlop"),
        ("DL::Seq_FlipFlop",  "DL::Seq_Counter"),
        ("DL::Seq_Counter",   "DL::Seq_FSM"),
    ]
    for p, d in dl_prereqs:
        dag.add_prerequisite(p, d)

    # ── Theory of Computation ──────────────────────────────────────────────
    toc_concepts = [
        ("TOC::RegEx",          "Theory of Computation", "Regular Languages",        "Regular Expressions"),
        ("TOC::DFA_NFA",        "Theory of Computation", "Regular Languages",        "DFA/NFA"),
        ("TOC::RegMinimize",    "Theory of Computation", "Regular Languages",        "Minimization"),
        ("TOC::CFG",            "Theory of Computation", "Context-Free Languages",   "CFG"),
        ("TOC::PDA",            "Theory of Computation", "Context-Free Languages",   "Pushdown Automata"),
        ("TOC::CFL_Properties", "Theory of Computation", "Context-Free Languages",   "CFL Properties"),
        ("TOC::TM_Basics",      "Theory of Computation", "Turing Machines",          "TM Basics"),
        ("TOC::Decidability",   "Theory of Computation", "Decidability",             "Decidable Problems"),
        ("TOC::PandNP",         "Theory of Computation", "Complexity Theory",        "P and NP"),
    ]
    for args in toc_concepts:
        dag.add_concept(*args)

    toc_prereqs = [
        ("TOC::RegEx",       "TOC::DFA_NFA"),
        ("TOC::DFA_NFA",     "TOC::RegMinimize"),
        ("TOC::DFA_NFA",     "TOC::CFG"),
        ("TOC::CFG",         "TOC::PDA"),
        ("TOC::PDA",         "TOC::CFL_Properties"),
        ("TOC::CFL_Properties", "TOC::TM_Basics"),
        ("TOC::TM_Basics",   "TOC::Decidability"),
        ("TOC::Decidability","TOC::PandNP"),
    ]
    for p, d in toc_prereqs:
        dag.add_prerequisite(p, d)

    # ── Mathematics ────────────────────────────────────────────────────────
    math_concepts = [
        ("MATH::PropLogic",    "Mathematics", "Propositional Logic",  "Logical Connectives"),
        ("MATH::PredLogic",    "Mathematics", "Predicate Logic",      "Quantifiers"),
        ("MATH::SetTheory",    "Mathematics", "Set Theory",           "Set Operations"),
        ("MATH::Relations",    "Mathematics", "Relations",            "Equivalence & Order"),
        ("MATH::Functions",    "Mathematics", "Functions",            "Bijections & Counting"),
        ("MATH::Combinatorics","Mathematics", "Combinatorics",        "Counting Principles"),
        ("MATH::Probability",  "Mathematics", "Probability",          "Basic Probability"),
        ("MATH::GraphTheory",  "Mathematics", "Graph Theory",         "Graph Properties"),
        ("MATH::LinearAlgebra","Mathematics", "Linear Algebra",       "Matrices"),
        ("MATH::Recurrence",   "Mathematics", "Recurrence Relations", "Solving Recurrences"),
        ("MATH::NumberTheory", "Mathematics", "Number Theory",        "Modular Arithmetic"),
    ]
    for args in math_concepts:
        dag.add_concept(*args)

    math_prereqs = [
        ("MATH::PropLogic",    "MATH::PredLogic"),
        ("MATH::SetTheory",    "MATH::Relations"),
        ("MATH::Relations",    "MATH::Functions"),
        ("MATH::Functions",    "MATH::Combinatorics"),
        ("MATH::Combinatorics","MATH::Probability"),
        ("MATH::SetTheory",    "MATH::GraphTheory"),
        ("MATH::PropLogic",    "MATH::Recurrence"),
    ]
    for p, d in math_prereqs:
        dag.add_prerequisite(p, d)

    # ── Programming & Data Structures ─────────────────────────────────────
    ds_concepts = [
        ("DS::Arrays",        "Programming and Data Structure", "Arrays & Strings",    "Array Operations"),
        ("DS::LinkedList",    "Programming and Data Structure", "Linked Lists",        "List Operations"),
        ("DS::StackQueue",    "Programming and Data Structure", "Stacks & Queues",     "Stack Applications"),
        ("DS::Trees_BST",     "Programming and Data Structure", "Trees",               "Binary Search Tree"),
        ("DS::Trees_Heap",    "Programming and Data Structure", "Trees",               "Heaps"),
        ("DS::Hashing",       "Programming and Data Structure", "Hashing",             "Hash Functions"),
        ("DS::Sorting",       "Programming and Data Structure", "Sorting",             "Comparison Sorts"),
        ("DS::Graphs",        "Programming and Data Structure", "Graphs",              "Graph Traversal"),
        ("DS::ShortestPath",  "Programming and Data Structure", "Graphs",              "Shortest Path"),
        ("DS::MST",           "Programming and Data Structure", "Graphs",              "Minimum Spanning Tree"),
        ("DS::Greedy",        "Programming and Data Structure", "Greedy Algorithms",   "Greedy Strategy"),
        ("DS::DP",            "Programming and Data Structure", "Dynamic Programming", "Knapsack"),
        ("DS::Huffman",       "Programming and Data Structure", "Compression",         "Huffman Coding"),
    ]
    for args in ds_concepts:
        dag.add_concept(*args)

    ds_prereqs = [
        ("DS::Arrays",     "DS::LinkedList"),
        ("DS::Arrays",     "DS::StackQueue"),
        ("DS::LinkedList", "DS::Trees_BST"),
        ("DS::StackQueue", "DS::Trees_BST"),
        ("DS::Trees_BST",  "DS::Trees_Heap"),
        ("DS::Trees_BST",  "DS::Graphs"),
        ("DS::Arrays",     "DS::Hashing"),
        ("DS::Arrays",     "DS::Sorting"),
        ("DS::Graphs",     "DS::ShortestPath"),
        ("DS::Graphs",     "DS::MST"),
        ("DS::Sorting",    "DS::Greedy"),
        ("DS::Trees_BST",  "DS::DP"),
        ("DS::Trees_Heap", "DS::Huffman"),
    ]
    for p, d in ds_prereqs:
        dag.add_prerequisite(p, d)

    # ── COA ────────────────────────────────────────────────────────────────
    coa_concepts = [
        ("COA::ISA_Format",    "Computer Organization and Architecture", "Instruction Set Architecture", "Instruction Formats"),
        ("COA::ISA_Addressing","Computer Organization and Architecture", "Instruction Set Architecture", "Addressing Modes"),
        ("COA::Pipeline",      "Computer Organization and Architecture", "Pipelining",                   "Pipeline Stages"),
        ("COA::Hazards",       "Computer Organization and Architecture", "Pipelining",                   "Hazards"),
        ("COA::Cache",         "Computer Organization and Architecture", "Memory Hierarchy",             "Cache Memory"),
        ("COA::CachePerf",     "Computer Organization and Architecture", "Memory Hierarchy",             "Cache Performance"),
        ("COA::Arithmetic",    "Computer Organization and Architecture", "Arithmetic",                   "Integer Arithmetic"),
        ("COA::FloatPoint",    "Computer Organization and Architecture", "Arithmetic",                   "Floating Point"),
        ("COA::Performance",   "Computer Organization and Architecture", "Performance",                  "CPU Performance"),
    ]
    for args in coa_concepts:
        dag.add_concept(*args)

    coa_prereqs = [
        ("COA::ISA_Format",    "COA::ISA_Addressing"),
        ("COA::ISA_Addressing","COA::Pipeline"),
        ("COA::Pipeline",      "COA::Hazards"),
        ("COA::ISA_Format",    "COA::Cache"),
        ("COA::Cache",         "COA::CachePerf"),
        ("COA::Arithmetic",    "COA::FloatPoint"),
        ("COA::Pipeline",      "COA::Performance"),
        ("COA::CachePerf",     "COA::Performance"),
    ]
    for p, d in coa_prereqs:
        dag.add_prerequisite(p, d)

    # ── General Aptitude ──────────────────────────────────────────────────
    ga_concepts = [
        ("GA::Arithmetic",  "General Aptitude", "Arithmetic",        "Profit & Loss"),
        ("GA::NumberSeries","General Aptitude", "Number Systems",    "Number Series"),
        ("GA::Algebra",     "General Aptitude", "Algebra",           "Equations"),
        ("GA::Logic",       "General Aptitude", "Logical Reasoning", "Puzzles"),
        ("GA::Verbal",      "General Aptitude", "Verbal Ability",    "Analogies"),
        ("GA::SetVenn",     "General Aptitude", "Set Theory Applied","Venn Diagrams"),
    ]
    for args in ga_concepts:
        dag.add_concept(*args)

    ga_prereqs = [
        ("GA::Arithmetic", "GA::Algebra"),
        ("GA::Algebra",    "GA::Logic"),
        ("GA::Arithmetic", "GA::SetVenn"),
    ]
    for p, d in ga_prereqs:
        dag.add_prerequisite(p, d)

    return dag
