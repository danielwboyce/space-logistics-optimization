# Class Diagram

Diagrams to show how classes in this code base interact

### Simplified Class Diagram

Main class methods you might use as a user are also included

```mermaid
classDiagram
OptModelBuilder --o Optimizer
OptModelBuilder : + build_model()
InitMixin --|> ComponentDesigner
InitMixin --|> NetworkBuilder
InitMixin --|> OptModelBuilder
InitMixin --|> Optimizer
InitMixin : +__init__()
InputData --o ComponentDesigner
InputData --o NetworkBuilder
InputData --o OptModelBuilder
InputData --o Optimizer
InputData --o InitMixin
InputData --o SpaceLogistics
ComponentDesigner --o OptModelBuilder
ComponentDesigner : + sc_sizing()
ComponentDesigner : + isru_des()
NetworkBuilder --o OptModelBuilder
NetworkBuilder --o Optimizer
ComponentDesigner --o Optimizer
ADMMLoop --* Optimizer
ADMMLoop : + run_alc_loop()
ADMMLoop : + solve_alc_subprob()
PWLApproximation --* Optimizer
PWLApproximation : + solve_w_pwl_approx()
FixedSCDesign --* Optimizer
FixedSCDesign : + solve_network_flow_MILP()
SolverInterface --* Optimizer
SolverInterface : + solve_model()
OuterLoop--* ADMMLoop
InnerLoop--* ADMMLoop
ComponentDesigner --o SpaceLogistics
NetworkBuilder --o SpaceLogistics
OptModelBuilder --o SpaceLogistics
Optimizer --o SpaceLogistics
```

### Full Class Diagram

```mermaid
classDiagram
MissonParameters --o InputData
ALCParameters --o InputData
SCParameters --o InputData
ISRUParameters --o InputData
CommodityDetails --o InputData
NodeDetails --o InputData
RuntimeSettings --o InputData
ScenarioDistribution --o InputData
OptModelBuilder --o Optimizer
OptModelBuilder : + build_model()
InitMixin --|> ComponentDesigner
InitMixin --|> NetworkBuilder
InitMixin --|> OptModelBuilder
InitMixin --|> Optimizer
InitMixin : +__init__()
InputData --o ComponentDesigner
InputData --o NetworkBuilder
InputData --o OptModelBuilder
InputData --o Optimizer
InputData --o InitMixin
InputData --o SpaceLogistics
SCSizing --* ComponentDesigner
ISRUDesign --* ComponentDesigner
ComponentDesigner --o OptModelBuilder
ComponentDesigner : + sc_sizing()
ComponentDesigner : + isru_des()
NetworkBuilder --o OptModelBuilder
NetworkBuilder --o Optimizer
Indices --* OptModelBuilder
Variables --* OptModelBuilder
Constraints --* OptModelBuilder
Objective --* OptModelBuilder
ComponentDesigner --o Optimizer
ADMMLoop --* Optimizer
ADMMLoop : + run_alc_loop()
ADMMLoop : + solve_alc_subprob()
PWLApproximation --* Optimizer
PWLApproximation : + solve_w_pwl_approx()
FixedSCDesign --* Optimizer
FixedSCDesign : + solve_network_flow_MILP()
SolverInterface --* Optimizer
SolverInterface : + solve_model()
OutputManager --* Optimizer
OutputManager : write_results()
OuterLoop--* ADMMLoop
InnerLoop--* ADMMLoop
SCsizingPygmo--* Optimizer
ComponentDesigner --o SpaceLogistics
NetworkBuilder --o SpaceLogistics
OptModelBuilder --o SpaceLogistics
Optimizer --o SpaceLogistics
Visualizer --o SpaceLogistics
```

Class diagram notations are as follows:

```mermaid
classDiagram
classA --|> classB : Inheritance
classC --* classD : Composition
classE --o classF : Aggregation
```

where

- classB inherits classA
- classD composes (i.e., owns) classC
- classF aggregates (i.e., contains) classE

See [inheritance vs composition vs aggregation](https://dev.to/adhirajk/inheritance-vs-composition-vs-aggregation-432i) for your reference. In short, an aggregated class can exist without its aggregating class but a composed class cannot without its composing class (stronger dependency).
