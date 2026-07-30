Functions:

Classes:

Program:
  (
    'decl',
    'hbar',
    'ℝ',
    ('num', 1.0),
    2,
  )
  (
    'decl',
    'mass',
    'ℝ',
    ('num', 1.0),
    3,
  )
  (
    'decl',
    'angular_frequency',
    'ℝ',
    ('num', 1.0),
    4,
  )
  (
    'decl',
    'pi',
    'ℝ',
    ('num', 3.141592653589793),
    5,
  )
  (
    'decl',
    'N_levels',
    'ℕ',
    ('num', 5),
    8,
  )
  (
    'decl',
    'x_max',
    'ℝ',
    ('num', 6.0),
    9,
  )
  (
    'decl',
    'N_grid',
    'ℕ',
    ('num', 601),
    10,
  )
  (
    'decl',
    'dx',
    'ℝ',
    (
      'div',
      (
        'mul',
        ('num', 2.0),
        ('var', 'x_max'),
      ),
      (
        'sub',
        ('var', 'N_grid'),
        ('num', 1),
      ),
    ),
    11,
  )
  (
    'decl',
    'position',
    (
      'tensor',
      [
        ('N_grid', 'invariant'),
      ],
    ),
    (
      'for_expr',
      'i',
      ('var', 'N_grid'),
      (
        'add',
        (
          'neg',
          ('var', 'x_max'),
        ),
        (
          'mul',
          ('imaginary'),
          ('var', 'dx'),
        ),
      ),
    ),
    14,
  )
  (
    'decl',
    'potential',
    (
      'tensor',
      [
        ('N_grid', 'invariant'),
      ],
    ),
    (
      'for_expr',
      'i',
      ('var', 'N_grid'),
      (
        'mul',
        (
          'mul',
          (
            'mul',
            (
              'mul',
              (
                'mul',
                ('num', 0.5),
                ('var', 'mass'),
              ),
              ('var', 'angular_frequency'),
            ),
            ('var', 'angular_frequency'),
          ),
          (
            'index',
            'position',
            ('imaginary'),
          ),
        ),
        (
          'index',
          'position',
          ('imaginary'),
        ),
      ),
    ),
    15,
  )
  (
    'decl',
    'wavefunctions',
    (
      'tensor',
      [
        ('N_levels', 'invariant'),
        ('N_grid', 'invariant'),
      ],
    ),
    (
      'for_expr',
      'n',
      ('var', 'N_levels'),
      (
        'for_expr',
        'i',
        ('var', 'N_grid'),
        (
          'mul',
          (
            'add',
            ('var', 'n'),
            ('imaginary'),
          ),
          ('num', 0.0),
        ),
      ),
    ),
    19,
  )
  (
    'decl',
    'hamiltonian_wavefunctions',
    (
      'tensor',
      [
        ('N_levels', 'invariant'),
        ('N_grid', 'invariant'),
      ],
    ),
    (
      'for_expr',
      'n',
      ('var', 'N_levels'),
      (
        'for_expr',
        'i',
        ('var', 'N_grid'),
        (
          'mul',
          (
            'add',
            ('var', 'n'),
            ('imaginary'),
          ),
          ('num', 0.0),
        ),
      ),
    ),
    20,
  )
  (
    'decl',
    'energies',
    (
      'tensor',
      [
        ('N_levels', 'invariant'),
      ],
    ),
    (
      'for_expr',
      'n',
      ('var', 'N_levels'),
      (
        'mul',
        ('var', 'n'),
        ('num', 0.0),
      ),
    ),
    22,
  )
  (
    'decl',
    'x',
    'ℝ',
    ('num', 0.0),
    25,
  )
  (
    'decl',
    'gaussian',
    'ℝ',
    ('num', 0.0),
    26,
  )
  (
    'decl',
    'n_real',
    'ℝ',
    ('num', 0.0),
    27,
  )
  (
    'decl',
    'next_n_real',
    'ℝ',
    ('num', 0.0),
    28,
  )
  (
    'decl',
    'first_coefficient',
    'ℝ',
    ('num', 0.0),
    29,
  )
  (
    'decl',
    'second_coefficient',
    'ℝ',
    ('num', 0.0),
    30,
  )
  (
    'decl',
    'second_derivative',
    'ℝ',
    ('num', 0.0),
    31,
  )
  (
    'decl',
    'kinetic_part',
    'ℝ',
    ('num', 0.0),
    32,
  )
  (
    'decl',
    'potential_part',
    'ℝ',
    ('num', 0.0),
    33,
  )
  (
    'decl',
    'energy_numerator',
    'ℝ',
    ('num', 0.0),
    34,
  )
  (
    'decl',
    'normalization_integral',
    'ℝ',
    ('num', 0.0),
    35,
  )
  (
    'decl',
    'normalization_constant',
    'ℝ',
    (
      'div',
      ('num', 1.0),
      (
        'pow',
        ('var', 'pi'),
        ('num', 0.25),
      ),
    ),
    38,
  )
  (
    'for_loop_range',
    'i',
    ('num', 0),
    ('var', 'N_grid'),
    [
      (
        'for_assign',
        'x',
        (
          'index',
          'position',
          ('imaginary'),
        ),
      ),
      (
        'for_assign',
        'gaussian',
        (
          'call',
          'exp',
          [
            (
              'mul',
              (
                'mul',
                (
                  'neg',
                  ('num', 0.5),
                ),
                ('var', 'x'),
              ),
              ('var', 'x'),
            ),
          ],
        ),
      ),
      (
        'for_index_assign_nd',
        'wavefunctions',
        [
          (
            'index_item',
            ('num', 0),
          ),
          (
            'index_item',
            ('imaginary'),
          ),
        ],
        (
          'mul',
          ('var', 'normalization_constant'),
          ('var', 'gaussian'),
        ),
      ),
    ],
    39,
  )
  (
    'decl',
    'one_level',
    'ℕ',
    ('num', 1),
    45,
  )
  (
    'if_only',
    (
      'cond_gt',
      ('var', 'N_levels'),
      ('var', 'one_level'),
    ),
    [
      (
        'for_loop_range',
        'i',
        ('num', 0),
        ('var', 'N_grid'),
        [
          (
            'for_assign',
            'x',
            (
              'index',
              'position',
              ('imaginary'),
            ),
          ),
          (
            'for_index_assign_nd',
            'wavefunctions',
            [
              (
                'index_item',
                ('num', 1),
              ),
              (
                'index_item',
                ('imaginary'),
              ),
            ],
            (
              'mul',
              (
                'mul',
                (
                  'call',
                  'sqrt',
                  [
                    ('num', 2.0),
                  ],
                ),
                ('var', 'x'),
              ),
              (
                'indexN',
                'wavefunctions',
                [
                  (
                    'index_item',
                    ('num', 0),
                  ),
                  (
                    'index_item',
                    ('imaginary'),
                  ),
                ],
              ),
            ),
          ),
        ],
        47,
      ),
    ],
  )
  (
    'decl',
    'two_levels',
    'ℕ',
    ('num', 2),
    52,
  )
  (
    'if_only',
    (
      'cond_gt',
      ('var', 'N_levels'),
      ('var', 'two_levels'),
    ),
    [
      (
        'for_loop_range',
        'n',
        ('num', 1),
        (
          'sub',
          ('var', 'N_levels'),
          ('num', 1),
        ),
        [
          (
            'for_assign',
            'n_real',
            (
              'mul',
              ('var', 'n'),
              ('num', 1.0),
            ),
          ),
          (
            'for_assign',
            'next_n_real',
            (
              'add',
              ('var', 'n_real'),
              ('num', 1.0),
            ),
          ),
          (
            'for_assign',
            'first_coefficient',
            (
              'call',
              'sqrt',
              [
                (
                  'div',
                  ('num', 2.0),
                  ('var', 'next_n_real'),
                ),
              ],
            ),
          ),
          (
            'for_assign',
            'second_coefficient',
            (
              'call',
              'sqrt',
              [
                (
                  'div',
                  ('var', 'n_real'),
                  ('var', 'next_n_real'),
                ),
              ],
            ),
          ),
          (
            'for_loop_range',
            'i',
            ('num', 0),
            ('var', 'N_grid'),
            [
              (
                'for_assign',
                'x',
                (
                  'index',
                  'position',
                  ('imaginary'),
                ),
              ),
              (
                'for_index_assign_nd',
                'wavefunctions',
                [
                  (
                    'index_item',
                    (
                      'add',
                      ('var', 'n'),
                      ('num', 1),
                    ),
                  ),
                  (
                    'index_item',
                    ('imaginary'),
                  ),
                ],
                (
                  'sub',
                  (
                    'mul',
                    (
                      'mul',
                      ('var', 'first_coefficient'),
                      ('var', 'x'),
                    ),
                    (
                      'indexN',
                      'wavefunctions',
                      [
                        (
                          'index_item',
                          ('var', 'n'),
                        ),
                        (
                          'index_item',
                          ('imaginary'),
                        ),
                      ],
                    ),
                  ),
                  (
                    'mul',
                    ('var', 'second_coefficient'),
                    (
                      'indexN',
                      'wavefunctions',
                      [
                        (
                          'index_item',
                          (
                            'sub',
                            ('var', 'n'),
                            ('num', 1),
                          ),
                        ),
                        (
                          'index_item',
                          ('imaginary'),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ],
            59,
          ),
        ],
        54,
      ),
    ],
  )
  (
    'for_loop_range',
    'n',
    ('num', 0),
    ('var', 'N_levels'),
    [
      (
        'for_loop_range',
        'i',
        ('num', 1),
        (
          'sub',
          ('var', 'N_grid'),
          ('num', 1),
        ),
        [
          (
            'for_assign',
            'second_derivative',
            (
              'div',
              (
                'add',
                (
                  'sub',
                  (
                    'indexN',
                    'wavefunctions',
                    [
                      (
                        'index_item',
                        ('var', 'n'),
                      ),
                      (
                        'index_item',
                        (
                          'add',
                          ('imaginary'),
                          ('num', 1),
                        ),
                      ),
                    ],
                  ),
                  (
                    'mul',
                    ('num', 2.0),
                    (
                      'indexN',
                      'wavefunctions',
                      [
                        (
                          'index_item',
                          ('var', 'n'),
                        ),
                        (
                          'index_item',
                          ('imaginary'),
                        ),
                      ],
                    ),
                  ),
                ),
                (
                  'indexN',
                  'wavefunctions',
                  [
                    (
                      'index_item',
                      ('var', 'n'),
                    ),
                    (
                      'index_item',
                      (
                        'sub',
                        ('imaginary'),
                        ('num', 1),
                      ),
                    ),
                  ],
                ),
              ),
              (
                'mul',
                ('var', 'dx'),
                ('var', 'dx'),
              ),
            ),
          ),
          (
            'for_assign',
            'kinetic_part',
            (
              'mul',
              (
                'div',
                (
                  'neg',
                  (
                    'mul',
                    ('var', 'hbar'),
                    ('var', 'hbar'),
                  ),
                ),
                (
                  'mul',
                  ('num', 2.0),
                  ('var', 'mass'),
                ),
              ),
              ('var', 'second_derivative'),
            ),
          ),
          (
            'for_assign',
            'potential_part',
            (
              'mul',
              (
                'index',
                'potential',
                ('imaginary'),
              ),
              (
                'indexN',
                'wavefunctions',
                [
                  (
                    'index_item',
                    ('var', 'n'),
                  ),
                  (
                    'index_item',
                    ('imaginary'),
                  ),
                ],
              ),
            ),
          ),
          (
            'for_index_assign_nd',
            'hamiltonian_wavefunctions',
            [
              (
                'index_item',
                ('var', 'n'),
              ),
              (
                'index_item',
                ('imaginary'),
              ),
            ],
            (
              'add',
              ('var', 'kinetic_part'),
              ('var', 'potential_part'),
            ),
          ),
        ],
        66,
      ),
    ],
    65,
  )
  (
    'for_loop_range',
    'n',
    ('num', 0),
    ('var', 'N_levels'),
    [
      (
        'for_assign',
        'energy_numerator',
        ('num', 0.0),
      ),
      (
        'for_assign',
        'normalization_integral',
        ('num', 0.0),
      ),
      (
        'for_loop_range',
        'i',
        ('num', 1),
        (
          'sub',
          ('var', 'N_grid'),
          ('num', 1),
        ),
        [
          (
            'for_assign',
            'energy_numerator',
            (
              'add',
              ('var', 'energy_numerator'),
              (
                'mul',
                (
                  'mul',
                  (
                    'indexN',
                    'wavefunctions',
                    [
                      (
                        'index_item',
                        ('var', 'n'),
                      ),
                      (
                        'index_item',
                        ('imaginary'),
                      ),
                    ],
                  ),
                  (
                    'indexN',
                    'hamiltonian_wavefunctions',
                    [
                      (
                        'index_item',
                        ('var', 'n'),
                      ),
                      (
                        'index_item',
                        ('imaginary'),
                      ),
                    ],
                  ),
                ),
                ('var', 'dx'),
              ),
            ),
          ),
          (
            'for_assign',
            'normalization_integral',
            (
              'add',
              ('var', 'normalization_integral'),
              (
                'mul',
                (
                  'mul',
                  (
                    'indexN',
                    'wavefunctions',
                    [
                      (
                        'index_item',
                        ('var', 'n'),
                      ),
                      (
                        'index_item',
                        ('imaginary'),
                      ),
                    ],
                  ),
                  (
                    'indexN',
                    'wavefunctions',
                    [
                      (
                        'index_item',
                        ('var', 'n'),
                      ),
                      (
                        'index_item',
                        ('imaginary'),
                      ),
                    ],
                  ),
                ),
                ('var', 'dx'),
              ),
            ),
          ),
        ],
        77,
      ),
      (
        'for_index_assign_nd',
        'energies',
        [
          (
            'index_item',
            ('var', 'n'),
          ),
        ],
        (
          'div',
          ('var', 'energy_numerator'),
          ('var', 'normalization_integral'),
        ),
      ),
      (
        'for_call',
        'physika_print',
        [
          (
            'index',
            'energies',
            ('var', 'n'),
          ),
        ],
      ),
    ],
    74,
  )
