from __future__ import annotations


class Node:
    """
    Represents a node in a computational graph for automatic differentiation.

    Each node stores a numerical value (data) and maintains connections to its
    parent nodes (children in the forward pass). This enables both forward
    computation and backward propagation of gradients.
    """

    def __init__(
        self,
        data: float,
        _children: tuple[Node, ...] = (),
        _op: str = "",
        label: str = "",
        _grad: float = 0.0,
    ) -> None:
        """
        Initialize a Node in the computational graph.

        Args:
            data: The numerical value stored in this node.
            _children: Tuple of child nodes (inputs to this node's operation).
                       Stored as a set internally for fast lookup and deduplication.
            _op: The operation that produced this node (e.g., "+", "*", "" for leaf nodes).
            label: Human-readable label for visualization purposes.
            _grad: The gradient of this node (used during backpropagation).
        """
        self.data = data
        # Convert children tuple to set for O(1) membership testing and deduplication.
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._grad = _grad

    def __repr__(self) -> str:
        """
        String representation of the node.

        Returns:
            A string showing the node's data value.
        """
        return f"Value(data={self.data})"

    def __add__(self, other: Node) -> Node:
        """
        Overload the addition operator to create a new node.

        Creates a new node representing the sum of this node and another node.
        The new node's children are set to (self, other) and its operation is "+".

        Args:
            other: The other node to add to this node.

        Returns:
            A new Node representing the sum of self and other.
        """
        return Node(self.data + other.data, (self, other), "+")

    def __mul__(self, other: Node) -> Node:
        """
        Overload the multiplication operator to create a new node.

        Creates a new node representing the product of this node and another node.
        The new node's children are set to (self, other) and its operation is "*".

        Args:
            other: The other node to multiply with this node.

        Returns:
            A new Node representing the product of self and other.
        """
        return Node(self.data * other.data, (self, other), "*")


if __name__ == "__main__":
    pass
