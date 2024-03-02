// plans.ts

export interface Plan {
  name: string;
  price: number;
  max_requests?: number;
}

export const plans: Plan[] = [
  {
    name: 'Free Plan',
    price: 9.99,
    max_requests: 50
  },
  {
    name: 'Basic Plan',
    price: 9.99,
    max_requests: 100
  },
  {
    name: 'Standard Plan',
    price: 19.99,
    max_requests: 200
  },
  {
    name: 'Premium Plan',
    price: 29.99,
    max_requests: 300
  }
];
